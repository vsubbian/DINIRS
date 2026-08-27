"""DINIRS model: encoder with survival attention gate, counterfactual generator,
discriminator, and doubly robust ITE predictor."""

import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vaswani et al. (NeurIPS 2017)."""

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """Base Transformer encoder for ICU multivariate time series."""

    def __init__(
        self,
        n_covariates,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        max_len=500,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_covariates = n_covariates

        self.input_proj = nn.Sequential(
            nn.Linear(n_covariates, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, pad_mask=None):
        """x: (batch, T, n_covariates) — raw time series"""
        h = self.input_proj(x)

        h = self.pos_encoder(h)

        attn_out = self.transformer(
            h,
            src_key_padding_mask=pad_mask,
        )

        if pad_mask is not None:
            valid_mask = (~pad_mask).unsqueeze(-1).float()
            emb = (attn_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            emb = attn_out.mean(dim=1)

        emb = self.output_proj(emb)

        return emb, attn_out


class SurvivalAttentionGate(nn.Module):
    """Learns a soft gate that separates survival-relevant from VFD-relevant"""

    def __init__(self, d_model, hidden_dim=64):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, emb):
        """emb: (batch, d_model) — patient embedding from encoder"""
        gate = self.gate_net(emb)
        emb_survival = emb * gate
        emb_vfd = emb * (1.0 - gate)

        return gate, emb_survival, emb_vfd


class SurvivalAwareTransformerEncoder(nn.Module):
    """Transformer encoder with survival-aware gating (NOVEL)."""

    def __init__(
        self,
        n_covariates,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        max_len=500,
        gate_hidden=64,
    ):
        super().__init__()

        self.base_encoder = TemporalTransformerEncoder(
            n_covariates=n_covariates,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        self.survival_gate = SurvivalAttentionGate(d_model, gate_hidden)

        self.d_model = d_model

    def forward(self, x, pad_mask=None):
        """x: (batch, T, n_covariates) — raw time series"""
        emb, attn_out = self.base_encoder(x, pad_mask)

        gate, emb_survival, emb_vfd = self.survival_gate(emb)

        return emb, emb_survival, emb_vfd, gate, attn_out


class CounterfactualGenerator(nn.Module):
    """Two-headed counterfactual generator for VFD-28 with survival decomposition."""

    def __init__(self, emb_dim=128, noise_dim=8, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.noise_dim = noise_dim

        input_dim = emb_dim + 1 + noise_dim

        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.vfd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, emb, treatment, noise):
        """Generate counterfactual outcomes for a given treatment."""
        h = torch.cat([emb, treatment, noise], dim=-1)

        h = self.shared_trunk(h)

        p_survive = self.survival_head(h)
        vfd_cond = self.vfd_head(h).clamp(max=28.0)

        vfd_composite = p_survive * vfd_cond

        return p_survive, vfd_cond, vfd_composite

    def forward_with_gated_emb(self, emb, treatment, noise, emb_survival, emb_vfd):
        """Generate outcomes using gated embeddings."""
        h = torch.cat([emb, treatment, noise], dim=-1)
        h = self.shared_trunk(h)

        h_surv = h + emb_survival[:, :h.size(-1)]
        p_survive = self.survival_head(h_surv)

        h_vfd = h + emb_vfd[:, :h.size(-1)]
        vfd_cond = self.vfd_head(h_vfd).clamp(max=28.0)

        vfd_composite = p_survive * vfd_cond
        return p_survive, vfd_cond, vfd_composite

    def generate_counterfactuals(self, emb, noise,
                                  emb_survival=None, emb_vfd=None):
        """Generate outcomes under BOTH treatments (for discriminator training)."""
        batch_size = emb.size(0)
        device = emb.device

        t0 = torch.zeros(batch_size, 1, device=device)
        t1 = torch.ones(batch_size, 1, device=device)

        if emb_survival is not None and emb_vfd is not None:
            p_surv_0, vfd_cond_0, vfd_0 = self.forward_with_gated_emb(
                emb, t0, noise, emb_survival, emb_vfd)
            p_surv_1, vfd_cond_1, vfd_1 = self.forward_with_gated_emb(
                emb, t1, noise, emb_survival, emb_vfd)
        else:
            p_surv_0, vfd_cond_0, vfd_0 = self.forward(emb, t0, noise)
            p_surv_1, vfd_cond_1, vfd_1 = self.forward(emb, t1, noise)

        return {
            'p_surv_0': p_surv_0, 'vfd_cond_0': vfd_cond_0, 'vfd_0': vfd_0,
            'p_surv_1': p_surv_1, 'vfd_cond_1': vfd_cond_1, 'vfd_1': vfd_1,
        }


class TreatmentDiscriminator(nn.Module):
    """Discriminator that receives patient embedding + generated outcomes"""

    def __init__(self, emb_dim=128, hidden_dim=128):
        super().__init__()
        input_dim = emb_dim + 6

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 4, 1)),
            nn.Sigmoid(),
        )

    def forward(self, emb, outcomes):
        """emb: (batch, emb_dim) — patient embedding"""
        h = torch.cat([emb, outcomes], dim=-1)
        return self.net(h)

    def gradient_penalty(self, emb, real_outcomes, fake_outcomes, lambda_gp=10.0):
        """Gradient penalty for improved training stability."""
        batch_size = real_outcomes.size(0)
        device = real_outcomes.device

        alpha = torch.rand(batch_size, 1, device=device)

        interpolated = (alpha * real_outcomes + (1 - alpha) * fake_outcomes)
        interpolated.requires_grad_(True)

        d_interpolated = self.forward(emb.detach(), interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gp = lambda_gp * ((gradient_norm - 1.0) ** 2).mean()

        return gp


class ITEPredictor(nn.Module):
    """Inference-time ITE predictor with survival decomposition."""

    def __init__(self, emb_dim=128, hidden_dim=128, dropout=0.1, cov_dim=23):
        super().__init__()

        self.cov_dim = cov_dim
        self.joint_mode = True
        if cov_dim and cov_dim > 0:
            self.cov_proj = nn.Sequential(
                nn.Linear(cov_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.shared = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.surv_head_0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.surv_head_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.vfd_head_0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.vfd_head_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, emb, x_cov=None, tau_base=None):
        """emb: (batch, emb_dim) — patient embedding"""
        h = self.shared(emb)

        if x_cov is not None and getattr(self, 'cov_dim', 0):
            h = h + self.cov_proj(x_cov)

        p_surv_0 = self.surv_head_0(h)
        vfd_cond_0 = self.vfd_head_0(h).clamp(max=28.0)
        vfd_0 = p_surv_0 * vfd_cond_0

        p_surv_1 = self.surv_head_1(h)
        vfd_cond_1 = self.vfd_head_1(h).clamp(max=28.0)
        vfd_1 = p_surv_1 * vfd_cond_1

        ite_decomposed = vfd_1 - vfd_0

        if tau_base is None:
            base_hint = ite_decomposed
        else:
            base_hint = tau_base
        h_ite = torch.cat([h, base_hint.detach()], dim=-1)
        if getattr(self, 'joint_mode', True):
            correction = self.correction_head(h_ite)
            ite = ite_decomposed + correction
        else:
            base = ite_decomposed if tau_base is None else tau_base
            correction = self.correction_head(h_ite)
            ite = base + correction

        ite_survival = p_surv_1 - p_surv_0
        ite_vfd_cond = vfd_cond_1 - vfd_cond_0

        return {
            'p_surv_0': p_surv_0, 'vfd_cond_0': vfd_cond_0, 'vfd_0': vfd_0,
            'p_surv_1': p_surv_1, 'vfd_cond_1': vfd_cond_1, 'vfd_1': vfd_1,
            'ite': ite,
            'ite_correction': correction,
            'ite_decomposed': ite_decomposed,
            'ite_survival': ite_survival,
            'ite_vfd_cond': ite_vfd_cond,
        }


class DINIRSModel(nn.Module):
    """Virtual Twin for Non-Invasive Respiratory Support (DINIRS)."""

    def __init__(
        self,
        n_covariates=23,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        noise_dim=8,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.n_covariates = n_covariates
        self.n_cov_raw = n_covariates
        self.d_model = d_model
        self.noise_dim = noise_dim
        self.use_cov_fusion = True

        self.encoder = SurvivalAwareTransformerEncoder(
            n_covariates=n_covariates,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.generator = CounterfactualGenerator(
            emb_dim=d_model,
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.discriminator = TreatmentDiscriminator(
            emb_dim=d_model,
            hidden_dim=hidden_dim,
        )

        self.predictor = ITEPredictor(
            emb_dim=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            cov_dim=self.n_cov_raw,
        )

        self.propensity_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def cov_summary(self, x, pad_mask=None):
        """Masked time-average of the raw covariates -> (batch, n_cov_raw)."""
        x = x[..., :self.n_cov_raw]
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(-1).float()
            denom = valid.sum(dim=1).clamp(min=1.0)
            return (x * valid).sum(dim=1) / denom
        return x.mean(dim=1)

    def encode(self, x, pad_mask=None):
        """Encode patient time series into embeddings."""
        return self.encoder(x, pad_mask)

    def forward_generator(self, x, treatment, pad_mask=None, noise=None):
        """Stage 1 forward pass: Encoder → Generator (+ Propensity Head)."""
        batch_size = x.size(0)
        device = x.device

        emb, emb_survival, emb_vfd, gate, attn_out = self.encode(x, pad_mask)

        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)

        gen_outputs = self.generator.generate_counterfactuals(
            emb, noise, emb_survival=emb_survival, emb_vfd=emb_vfd)

        propensity_logits = self.propensity_head(emb.detach())

        encoder_outputs = (emb, emb_survival, emb_vfd, gate, attn_out,
                          propensity_logits)

        return gen_outputs, encoder_outputs

    def forward_discriminator(self, emb, gen_outputs):
        """Stage 1 discriminator pass."""
        outcomes = torch.cat([
            gen_outputs['p_surv_0'], gen_outputs['vfd_cond_0'], gen_outputs['vfd_0'],
            gen_outputs['p_surv_1'], gen_outputs['vfd_cond_1'], gen_outputs['vfd_1'],
        ], dim=-1)

        return self.discriminator(emb.detach(), outcomes)

    def forward_predictor(self, x, pad_mask=None, tau_base=None):
        """Stage 2 / inference: Encoder → ITEPredictor."""
        emb, emb_survival, emb_vfd, gate, attn_out = self.encode(x, pad_mask)
        x_cov = (self.cov_summary(x[..., :self.n_cov_raw], pad_mask)
                 if getattr(self, 'use_cov_fusion', True) else None)
        pred_outputs = self.predictor(emb, x_cov=x_cov, tau_base=tau_base)

        propensity_logits = self.propensity_head(emb.detach())
        encoder_outputs = (emb, emb_survival, emb_vfd, gate, attn_out,
                          propensity_logits)

        return pred_outputs, encoder_outputs

    def predict_ite(self, x, pad_mask=None, tau_base=None):
        """Convenience method for inference: returns just the ITE."""
        pred_outputs, _ = self.forward_predictor(x, pad_mask, tau_base=tau_base)
        return pred_outputs['ite']

    def get_treatment_recommendation(self, x, pad_mask=None):
        """Returns treatment recommendation for each patient."""
        pred_outputs, _ = self.forward_predictor(x, pad_mask)
        ite = pred_outputs['ite']

        rec = (ite > 0).squeeze(-1).long()

        return rec, ite, pred_outputs
