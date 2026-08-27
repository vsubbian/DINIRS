"""Two-stage training pipeline and loss functions for DINIRS."""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import sys
from models.dinirs import DINIRSModel
from utils.metrics import compute_all_metrics
import torch.nn.functional as F


class CensoringAwareAdversarialLoss(nn.Module):
    """Multi-objective loss for DINIRS training."""

    def __init__(
        self,
        lambda_adv=1.0,
        lambda_surv=1.0,
        lambda_vfd=1.0,
        lambda_consist=0.5,
        lambda_gate=0.1,
        lambda_ipm=1.0,
        lambda_dr=0.5,
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_surv = lambda_surv
        self.lambda_vfd = lambda_vfd
        self.lambda_consist = lambda_consist
        self.lambda_gate = lambda_gate
        self.lambda_ipm = lambda_ipm
        self.lambda_dr = lambda_dr

        self.bce = nn.BCELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='none')

    def adversarial_loss_generator(self, p_real_fake):
        """Generator wants discriminator to think fake outcomes are real."""
        target = torch.ones_like(p_real_fake)
        return self.bce(p_real_fake, target)

    def adversarial_loss_discriminator(self, p_real, p_fake):
        """Discriminator distinguishes real from generated outcomes."""
        real_loss = self.bce(p_real, torch.ones_like(p_real))
        fake_loss = self.bce(p_fake, torch.zeros_like(p_fake))
        return 0.5 * (real_loss + fake_loss)

    def survival_loss(self, p_survive_pred, delta, event_times=None):
        """Survival prediction loss with optional discrete-time extension."""
        bce_loss = self.bce(p_survive_pred, delta)

        if event_times is None:
            return bce_loss

        p_surv = p_survive_pred.squeeze()
        t = event_times.squeeze()
        d = delta.squeeze()

        n = p_surv.size(0)
        if n < 2:
            return bce_loss

        dead_mask = (d == 0)
        if dead_mask.sum() == 0 or (~dead_mask).sum() == 0:
            return bce_loss

        dead_p = p_surv[dead_mask]
        dead_t = t[dead_mask]
        alive_p = p_surv[~dead_mask]

        margin = 0.1
        n_pairs = min(dead_p.size(0), alive_p.size(0), 64)
        idx_dead = torch.randperm(dead_p.size(0))[:n_pairs]
        idx_alive = torch.randperm(alive_p.size(0))[:n_pairs]

        ranking_loss = torch.clamp(
            dead_p[idx_dead] - alive_p[idx_alive] + margin, min=0.0).mean()

        return bce_loss + 0.5 * ranking_loss

    def conditional_vfd_loss(self, vfd_cond_pred, vfd_observed, delta):
        """MSE for conditional VFD-28, computed ONLY for survivors."""
        per_sample_mse = self.mse(vfd_cond_pred, vfd_observed)

        masked_mse = per_sample_mse * delta

        n_survivors = delta.sum().clamp(min=1.0)
        return masked_mse.sum() / n_survivors

    def consistency_loss(self, gen_outputs, pred_outputs, observed_treatment):
        """Jensen-Shannon divergence between Generator and Predictor outputs"""
        t = observed_treatment.squeeze(-1)

        gen_vfd = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['vfd_1'],
            gen_outputs['vfd_0']
        )
        pred_vfd = torch.where(
            t.unsqueeze(-1) == 1,
            pred_outputs['vfd_1'],
            pred_outputs['vfd_0']
        )

        return F.mse_loss(pred_vfd, gen_vfd.detach())

    def gate_entropy_loss(self, gate):
        """Encourages the survival gate to be sharp (close to 0 or 1)."""
        eps = 1e-8
        entropy = -(gate * torch.log(gate + eps) +
                     (1 - gate) * torch.log(1 - gate + eps))
        return entropy.mean()

    def mmd_loss(self, emb_treated, emb_control, kernel='rbf', bandwidth=None):
        """Maximum Mean Discrepancy (MMD) between treated and control embeddings."""
        if emb_treated.size(0) == 0 or emb_control.size(0) == 0:
            return torch.tensor(0.0, device=emb_treated.device)

        if kernel == 'linear':
            mean_t = emb_treated.mean(dim=0)
            mean_c = emb_control.mean(dim=0)
            return ((mean_t - mean_c) ** 2).sum()

        all_emb = torch.cat([emb_treated, emb_control], dim=0)
        pairwise_dist = torch.cdist(all_emb, all_emb, p=2)
        if bandwidth is None:
            bandwidth = torch.median(pairwise_dist[pairwise_dist > 0]).detach()
            bandwidth = bandwidth.clamp(min=1e-6)

        gamma = 1.0 / (2.0 * bandwidth ** 2)

        n_t = emb_treated.size(0)
        n_c = emb_control.size(0)

        K_tt = torch.exp(-gamma * torch.cdist(emb_treated, emb_treated, p=2) ** 2)
        K_cc = torch.exp(-gamma * torch.cdist(emb_control, emb_control, p=2) ** 2)
        K_tc = torch.exp(-gamma * torch.cdist(emb_treated, emb_control, p=2) ** 2)

        mmd = (K_tt.sum() / (n_t * n_t)
               - 2.0 * K_tc.sum() / (n_t * n_c)
               + K_cc.sum() / (n_c * n_c))

        return mmd

    def propensity_loss(self, propensity_logits, observed_treatment):
        """Binary cross-entropy for propensity score estimation."""
        return F.binary_cross_entropy_with_logits(propensity_logits, observed_treatment)

    def compute_overlap_weights(self, propensity_scores, treatment):
        """Compute overlap weights that focus on the equipoise population."""
        e = propensity_scores.clamp(0.01, 0.99)
        w = treatment * (1 - e) + (1 - treatment) * e
        w = w * w.size(0) / w.sum().clamp(min=1e-6)
        return w.detach()

    def doubly_robust_loss(self, pred_ite, gen_outputs, propensity_scores,
                           observed_treatment, vfd_observed, delta,
                           ipcw_weight=None, prop_clip=0.1,
                           use_overlap_weights=True):
        """Doubly-robust (AIPW) pseudo-outcome loss for ITE predictor."""
        W = observed_treatment
        Y = delta * vfd_observed

        e = propensity_scores.clamp(prop_clip, 1.0 - prop_clip)

        if ipcw_weight is None:
             w_c = torch.ones_like(Y)
        else:
            w_c = ipcw_weight.clamp(min=1.0, max=20.0)

        mu_1 = gen_outputs['vfd_1'].detach()
        mu_0 = gen_outputs['vfd_0'].detach()

        dr_pseudo = (mu_1 - mu_0
                     + w_c * W / e * (Y - mu_1)
                     - w_c * (1 - W) / (1 - e) * (Y - mu_0))

        target = dr_pseudo.detach()
        if use_overlap_weights:
            ow = (W * (1 - e) + (1 - W) * e).detach()
            ow = ow * ow.numel() / ow.sum().clamp(min=1e-6)
            return (ow * (pred_ite - target) ** 2).mean()
        return F.mse_loss(pred_ite, target)

    def generator_loss(
        self,
        p_real_fake,
        gen_outputs,
        observed_treatment,
        vfd_observed,
        delta,
        gate,
        emb=None,
        propensity_logits=None,
    ):
        """Total Generator loss (Stage 1)."""
        t = observed_treatment.squeeze(-1)

        if p_real_fake is None:
            l_adv = torch.tensor(0.0, device=gate.device)
        else:
            l_adv = self.adversarial_loss_generator(p_real_fake)

        p_surv_obs = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['p_surv_1'],
            gen_outputs['p_surv_0']
        )
        l_surv = self.survival_loss(p_surv_obs, delta)

        vfd_cond_obs = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['vfd_cond_1'],
            gen_outputs['vfd_cond_0']
        )
        l_vfd = self.conditional_vfd_loss(vfd_cond_obs, vfd_observed, delta)

        l_gate = self.gate_entropy_loss(gate)

        l_mmd = torch.tensor(0.0, device=gate.device)
        if emb is not None:
            t_mask = (t == 1)
            c_mask = (t == 0)
            if t_mask.sum() > 0 and c_mask.sum() > 0:
                l_mmd = self.mmd_loss(emb[t_mask], emb[c_mask])

        l_prop = torch.tensor(0.0, device=gate.device)
        if propensity_logits is not None:
            l_prop = self.propensity_loss(propensity_logits, observed_treatment)

        total = (
            self.lambda_adv * l_adv
            + self.lambda_surv * l_surv
            + self.lambda_vfd * l_vfd
            + self.lambda_gate * l_gate
            + self.lambda_ipm * l_mmd
            + l_prop
        )

        return total, {
            'l_adv_G': l_adv.item(),
            'l_surv': l_surv.item(),
            'l_vfd': l_vfd.item(),
            'l_gate': l_gate.item(),
            'l_mmd': l_mmd.item(),
            'l_prop': l_prop.item(),
            'l_total_G': total.item(),
        }

    def predictor_loss(self, gen_outputs, pred_outputs, observed_treatment,
                        propensity_scores=None, vfd_observed=None, delta=None,
                        ipcw_weight=None, prop_clip=0.1,
                        use_overlap_weights=True, lambda_anchor=0.1,
                        lambda_fact=1.0,
                        lambda_joint=1.0):
        """Total Predictor loss (Stage 2)."""
        l_consist_obs = self.consistency_loss(gen_outputs, pred_outputs,
                                              observed_treatment)

        t = observed_treatment.squeeze(-1)
        gen_vfd_cf = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['vfd_0'],
            gen_outputs['vfd_1']
        )
        pred_vfd_cf = torch.where(
            t.unsqueeze(-1) == 1,
            pred_outputs['vfd_0'],
            pred_outputs['vfd_1']
        )
        l_consist_cf = F.mse_loss(pred_vfd_cf, gen_vfd_cf.detach())

        l_dr = torch.tensor(0.0, device=observed_treatment.device)
        if (propensity_scores is not None and vfd_observed is not None
                and delta is not None):
            l_dr = self.doubly_robust_loss(
                pred_outputs['ite'], gen_outputs, propensity_scores,
                observed_treatment, vfd_observed, delta,
                ipcw_weight=ipcw_weight, prop_clip=prop_clip,
                use_overlap_weights=use_overlap_weights)

        l_anchor = torch.tensor(0.0, device=observed_treatment.device)
        if 'ite_correction' in pred_outputs:
            l_anchor = (pred_outputs['ite_correction'] ** 2).mean()

        l_fact = torch.tensor(0.0, device=observed_treatment.device)
        if vfd_observed is not None and delta is not None:
            tt = observed_treatment.squeeze(-1).unsqueeze(-1)
            p_surv_obs = torch.where(tt == 1, pred_outputs['p_surv_1'],
                                     pred_outputs['p_surv_0'])
            vfd_cond_obs = torch.where(tt == 1, pred_outputs['vfd_cond_1'],
                                       pred_outputs['vfd_cond_0'])
            l_fact = (self.survival_loss(p_surv_obs, delta)
                      + self.conditional_vfd_loss(vfd_cond_obs, vfd_observed, delta))

        l_joint = torch.tensor(0.0, device=observed_treatment.device)
        if (lambda_joint > 0 and 'ite_decomposed' in pred_outputs
                and propensity_scores is not None and vfd_observed is not None
                and delta is not None):
            l_joint = self.doubly_robust_loss(
                pred_outputs['ite_decomposed'], gen_outputs, propensity_scores,
                observed_treatment, vfd_observed, delta,
                ipcw_weight=ipcw_weight, prop_clip=prop_clip,
                use_overlap_weights=use_overlap_weights)

        total = (lambda_fact * l_fact
                 + self.lambda_consist * l_consist_obs
                 + self.lambda_consist * l_consist_cf
                 + self.lambda_dr * l_dr
                 + lambda_anchor * l_anchor
                 + lambda_joint * l_joint)

        return total, {
            'l_fact': l_fact.item(),
            'l_consist_obs': l_consist_obs.item(),
            'l_consist_cf': l_consist_cf.item(),
            'l_dr': l_dr.item(),
            'l_anchor': l_anchor.item(),
            'l_joint': l_joint.item(),
            'l_consist': total.item(),
        }


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_CONFIG = {
    'n_covariates': 23,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 256,
    'noise_dim': 8,
    'hidden_dim': 128,
    'dropout': 0.1,

    'epochs_stage1': 100,
    'lr_generator': 2e-4,
    'lr_discriminator': 2e-4,
    'lr_encoder': 2e-4,
    'weight_decay': 1e-4,

    'epochs_stage2': 50,
    'lr_predictor': 2e-4,

    'lambda_adv': 1.0,
    'lambda_surv': 1.0,
    'lambda_vfd': 2.0,
    'lambda_consist': 1.0,
    'lambda_gate': 0.01,
    'lambda_ipm': 0.05,
    'lambda_dr': 1.0,
    'lambda_gp': 10.0,
    'prop_clip': 0.1,
    'use_overlap_weights': True,
    'lambda_anchor': 0.1,
    'lambda_fact': 1.0,
    'use_adversarial': True,
    'lambda_joint': 1.0,
    'min_epochs_stage1': 60,
    'select_on_c_for_benefit': True,
    'n_folds': 5,
    'seeds': (0, 1, 2, 3, 4),

    'batch_size': 128,
    'random_state': 42,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints/',
    'patience': 10,
}


def train_stage1(model, train_loader, val_loader, loss_fn, config, save_dir):
    """Stage 1: Adversarial training of Encoder + Generator + Discriminator."""
    os.makedirs(save_dir, exist_ok=True)
    device = config['device']
    model = model.to(device)

    params_gen = (list(model.encoder.parameters())
                  + list(model.generator.parameters())
                  + list(model.propensity_head.parameters()))
    params_disc = list(model.discriminator.parameters())

    opt_gen = torch.optim.Adam(params_gen, lr=config['lr_generator'],
                                weight_decay=config['weight_decay'])
    opt_disc = torch.optim.Adam(params_disc, lr=config['lr_discriminator'],
                                 weight_decay=config['weight_decay'])

    scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_gen, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_disc, mode='min', factor=0.5, patience=5, verbose=True)

    train_log = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config['epochs_stage1'] + 1):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            x = batch['x'].to(device)
            treatment = batch['treatment'].to(device)
            vfd = batch['vfd'].to(device)
            delta = batch['delta'].to(device)
            pad_mask = batch['pad_mask'].to(device)

            bs = x.size(0)
            noise = torch.randn(bs, config['noise_dim'], device=device)

            gen_outputs, enc_outputs = model.forward_generator(
                x, treatment, pad_mask, noise)
            emb, emb_surv, emb_vfd, gate, attn_out, prop_logits = enc_outputs

            use_adv = config.get('use_adversarial', True)
            if not use_adv:
                loss_D = torch.tensor(0.0, device=device)
                gp = torch.tensor(0.0, device=device)
                p_fake_for_G = None
            else:
                opt_disc.zero_grad()

                real_outcomes = _build_real_outcomes(gen_outputs, treatment, vfd, delta)
                p_real = model.forward_discriminator(emb, real_outcomes)

                p_fake = model.forward_discriminator(emb, gen_outputs)

                loss_D = loss_fn.adversarial_loss_discriminator(p_real, p_fake)

                real_out_tensor = torch.cat([
                    real_outcomes['p_surv_0'], real_outcomes['vfd_cond_0'],
                    real_outcomes['vfd_0'], real_outcomes['p_surv_1'],
                    real_outcomes['vfd_cond_1'], real_outcomes['vfd_1']
                ], dim=-1)
                fake_out_tensor = torch.cat([
                    gen_outputs['p_surv_0'], gen_outputs['vfd_cond_0'],
                    gen_outputs['vfd_0'], gen_outputs['p_surv_1'],
                    gen_outputs['vfd_cond_1'], gen_outputs['vfd_1']
                ], dim=-1)
                gp = model.discriminator.gradient_penalty(
                    emb, real_out_tensor.detach(), fake_out_tensor.detach(),
                    lambda_gp=config.get('lambda_gp', 10.0))

                (loss_D + gp).backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(params_disc, max_norm=1.0)
                opt_disc.step()

            opt_gen.zero_grad()

            if use_adv:
                p_fake_for_G = model.forward_discriminator(emb, gen_outputs)

            loss_G, loss_dict = loss_fn.generator_loss(
                p_real_fake=p_fake_for_G,
                gen_outputs=gen_outputs,
                observed_treatment=treatment,
                vfd_observed=vfd,
                delta=delta,
                gate=gate,
                emb=emb,
                propensity_logits=prop_logits,
            )
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_gen, max_norm=1.0)
            opt_gen.step()

            epoch_losses['l_D'] += loss_D.item()
            epoch_losses['l_gp'] += gp.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        model.eval()
        val_losses = defaultdict(float)
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                treatment = batch['treatment'].to(device)
                vfd = batch['vfd'].to(device)
                delta = batch['delta'].to(device)
                pad_mask = batch['pad_mask'].to(device)
                bs = x.size(0)
                noise = torch.randn(bs, config['noise_dim'], device=device)

                gen_outputs, enc_outputs = model.forward_generator(
                    x, treatment, pad_mask, noise)
                emb, _, _, gate, _, prop_logits = enc_outputs

                p_fake = model.forward_discriminator(emb, gen_outputs)

                _, loss_dict = loss_fn.generator_loss(
                    p_fake, gen_outputs, treatment, vfd, delta, gate,
                    emb=emb, propensity_logits=prop_logits)

                for k, v in loss_dict.items():
                    val_losses[k] += v
                n_val += 1

        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        train_log['epoch'].append(epoch)
        for k in epoch_losses:
            train_log[f'train_{k}'].append(epoch_losses[k])
        for k in val_losses:
            train_log[f'val_{k}'].append(val_losses[k])

        val_total = val_losses.get('l_total_G', float('inf'))
        scheduler_gen.step(val_total)
        scheduler_disc.step(val_total)

        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_stage1.pth'))
        else:
            patience_counter += 1

        if (epoch >= config.get('min_epochs_stage1', 60)
                and patience_counter >= config['patience']):
            break

    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_stage1.pth'),
                    map_location=device))

    return model, dict(train_log)


def train_stage2(model, train_loader, val_loader, loss_fn, config, save_dir):
    """Stage 2: Train ITEPredictor on Generator's pseudo-labels."""
    os.makedirs(save_dir, exist_ok=True)
    device = config['device']
    model = model.to(device)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.generator.parameters():
        p.requires_grad = False
    for p in model.discriminator.parameters():
        p.requires_grad = False

    opt_pred = torch.optim.Adam(
        model.predictor.parameters(),
        lr=config['lr_predictor'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_pred, mode='min', factor=0.5, patience=5, verbose=True)

    train_log = defaultdict(list)
    best_val_loss = float('inf')
    best_val_metric = -np.inf
    best_epoch = 0

    for epoch in range(1, config['epochs_stage2'] + 1):
        model.predictor.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch['x'].to(device)
            treatment = batch['treatment'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            vfd = batch['vfd'].to(device)
            delta = batch['delta'].to(device)
            bs = x.size(0)

            opt_pred.zero_grad()

            with torch.no_grad():
                noise = torch.randn(bs, config['noise_dim'], device=device)
                gen_outputs, enc_outputs = model.forward_generator(
                    x, treatment, pad_mask, noise)
                emb = enc_outputs[0]
                prop_logits = enc_outputs[5]
                prop_scores = torch.sigmoid(prop_logits)

            x_cov = model.cov_summary(x, pad_mask) if getattr(model, 'use_cov_fusion', True) else None
            tb = batch.get('tau_base', None)
            if tb is not None:
                tb = tb.to(device)
            pred_outputs = model.predictor(emb, x_cov=x_cov, tau_base=tb)

            loss, _ = loss_fn.predictor_loss(
                gen_outputs, pred_outputs, treatment,
                propensity_scores=prop_scores,
                vfd_observed=vfd,
                delta=delta,
                ipcw_weight=batch.get('ipcw', None),
                prop_clip=config.get('prop_clip', 0.1),
                use_overlap_weights=config.get('use_overlap_weights', True),
                lambda_anchor=config.get('lambda_anchor', 0.1),
                lambda_fact=config.get('lambda_fact', 1.0),
                lambda_joint=config.get('lambda_joint', 1.0),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            opt_pred.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        model.predictor.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                treatment = batch['treatment'].to(device)
                pad_mask = batch['pad_mask'].to(device)
                vfd = batch['vfd'].to(device)
                delta = batch['delta'].to(device)
                bs = x.size(0)

                noise = torch.randn(bs, config['noise_dim'], device=device)
                gen_outputs, enc_outputs = model.forward_generator(
                    x, treatment, pad_mask, noise)
                _xc = model.cov_summary(x, pad_mask) if getattr(model, 'use_cov_fusion', True) else None
                _tb = batch.get('tau_base', None)
                if _tb is not None:
                    _tb = _tb.to(device)
                pred_outputs = model.predictor(enc_outputs[0], x_cov=_xc, tau_base=_tb)
                prop_scores = torch.sigmoid(enc_outputs[5])
                loss, _ = loss_fn.predictor_loss(
                    gen_outputs, pred_outputs, treatment,
                    propensity_scores=prop_scores,
                    vfd_observed=vfd, delta=delta,
                    ipcw_weight=batch.get('ipcw', None),
                    prop_clip=config.get('prop_clip', 0.1),
                    use_overlap_weights=config.get('use_overlap_weights', True),
                    lambda_anchor=config.get('lambda_anchor', 0.1),
                    lambda_fact=config.get('lambda_fact', 1.0),
                    lambda_joint=config.get('lambda_joint', 1.0))
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        val_metric = None
        if config.get('select_on_c_for_benefit', True):
            try:
                from utils.metrics import c_for_benefit as _cfb
                _it, _vf, _tr = [], [], []
                model.predictor.eval()
                with torch.no_grad():
                    for b in val_loader:
                        _x = b['x'].to(device); _pm = b['pad_mask'].to(device)
                        _tb = b.get('tau_base')
                        _tb = _tb.to(device) if _tb is not None else None
                        _o, _ = model.forward_predictor(_x, _pm, tau_base=_tb)
                        _it.append(_o['ite'].cpu().numpy().ravel())
                        _vf.append(b['vfd'].numpy().ravel())
                        _tr.append(b['treatment'].numpy().ravel())
                _it = np.concatenate(_it); _vf = np.concatenate(_vf); _tr = np.concatenate(_tr)
                if len(np.unique(_tr)) > 1:
                    val_metric = _cfb(_it, _vf, _tr, random_state=0)['c_for_benefit']
            except Exception:
                val_metric = None

        train_log['epoch'].append(epoch)
        train_log['train_l_consist'].append(epoch_loss)
        train_log['val_l_consist'].append(val_loss)

        if val_metric is not None:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir, 'best_stage2.pth'))
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_stage2.pth'))

    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_stage2.pth'),
                    map_location=device))
    for p in model.parameters():
        p.requires_grad = True

    return model, dict(train_log)


def evaluate(model, test_loader, config):
    """Full evaluation on test set."""
    device = config['device']
    model = model.to(device)
    model.eval()

    all_preds = defaultdict(list)
    all_vfd = []
    all_delta = []
    all_treatment = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            pad_mask = batch['pad_mask'].to(device)

            _tb = batch.get('tau_base', None)
            if _tb is not None:
                _tb = _tb.to(device)
            pred_outputs, _ = model.forward_predictor(x, pad_mask, tau_base=_tb)

            for k, v in pred_outputs.items():
                all_preds[k].append(v.cpu().numpy())
            all_vfd.append(batch['vfd'].numpy())
            all_delta.append(batch['delta'].numpy())
            all_treatment.append(batch['treatment'].numpy())

    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)
    vfd_obs = np.concatenate(all_vfd, axis=0).squeeze()
    delta = np.concatenate(all_delta, axis=0).squeeze()
    treatment = np.concatenate(all_treatment, axis=0).squeeze()

    metrics = compute_all_metrics(all_preds, vfd_obs, delta, treatment)

    return metrics, all_preds


def _build_real_outcomes(gen_outputs, treatment, vfd_observed, delta):
    """Build "real" outcome tensor for discriminator training."""
    t = treatment.squeeze(-1)

    real_p_surv = delta
    real_vfd_cond = vfd_observed
    real_vfd = delta * vfd_observed

    p_surv_0 = torch.where(t.unsqueeze(-1) == 0, real_p_surv, gen_outputs['p_surv_0'])
    vfd_cond_0 = torch.where(t.unsqueeze(-1) == 0, real_vfd_cond, gen_outputs['vfd_cond_0'])
    vfd_0 = torch.where(t.unsqueeze(-1) == 0, real_vfd, gen_outputs['vfd_0'])

    p_surv_1 = torch.where(t.unsqueeze(-1) == 1, real_p_surv, gen_outputs['p_surv_1'])
    vfd_cond_1 = torch.where(t.unsqueeze(-1) == 1, real_vfd_cond, gen_outputs['vfd_cond_1'])
    vfd_1 = torch.where(t.unsqueeze(-1) == 1, real_vfd, gen_outputs['vfd_1'])

    return {
        'p_surv_0': p_surv_0, 'vfd_cond_0': vfd_cond_0, 'vfd_0': vfd_0,
        'p_surv_1': p_surv_1, 'vfd_cond_1': vfd_cond_1, 'vfd_1': vfd_1,
    }


def run_full_pipeline(train_loader, val_loader, test_loader, config=None):
    """Run the complete DINIRS training and evaluation pipeline."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    model = DINIRSModel(
        n_covariates=config['n_covariates'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        noise_dim=config['noise_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
    )

    total_params = sum(p.numel() for p in model.parameters())

    loss_fn = CensoringAwareAdversarialLoss(
        lambda_adv=config['lambda_adv'],
        lambda_surv=config['lambda_surv'],
        lambda_vfd=config['lambda_vfd'],
        lambda_consist=config['lambda_consist'],
        lambda_gate=config['lambda_gate'],
        lambda_ipm=config.get('lambda_ipm', 1.0),
        lambda_dr=config.get('lambda_dr', 0.5),
    )

    print('Stage 1 training...')
    model, log1 = train_stage1(model, train_loader, val_loader,
                                loss_fn, config, save_dir)

    print('Stage 2 training...')
    model, log2 = train_stage2(model, train_loader, val_loader,
                                loss_fn, config, save_dir)

    metrics, predictions = evaluate(model, test_loader, config)

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()}, f, indent=2)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                    for k, v in config.items()}, f, indent=2)

    train_logs = {'stage1': log1, 'stage2': log2}

    return model, metrics, train_logs
