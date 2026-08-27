# DINIRS: Digital Twin for Individualized Treatment Effects of Non-Invasive Respiratory Support

This repository provides the reproducible workflow for DINIRS, a censoring-aware digital twin framework that estimates individualized treatment effects (ITEs) of non-invasive respiratory support (NIRS) versus invasive mechanical ventilation (IMV) in acute respiratory failure. Models are developed on **MIMIC-IV v3.1** and externally validated on **eICU-CRD**. 

## 📁 Repository Structure

| File | Description |
| --- | --- |
| `DINIRS.ipynb` | Jupyter notebook implementing the end-to-end workflow: configuration, BigQuery extraction, cross-fitted training, evaluation, and figures. |
| `models/dinirs.py` | DINIRS model: survival encoder with attention gate, counterfactual generator, discriminator, and doubly robust ITE predictor. |
| `models/baselines.py` | Tree-based baselines (T-Learner, causal forest, causal survival forest) and the cross-fitted tree base for the doubly robust learner. |
| `models/ensemble.py` | Cross-validated ensemble producing the out-of-fold DINIRS estimate. |
| `training/train.py` | Two-stage training pipeline and loss functions. |
| `utils/extraction.py` | MIMIC-IV and eICU-CRD extraction, cohort construction, covariates, and matching. |
| `utils/metrics.py` | Evaluation metrics, doubly robust policy value, and sensitivity analyses. |
| `utils/generalization.py` | Cross-fitting, seed-stability, and subgroup evaluation. |
| `utils/mice.py` | Multiple imputation by chained equations with predictive mean matching. |
| `requirements.txt` | Python package requirements. |
| `README.md` | Project overview, setup instructions, and reproducibility notes. |

## 📦 Getting Started

To reproduce results, you can either work locally with PhysioNet files or query the hosted copy in Google BigQuery. Both paths are outlined below.

After you complete the credentialed-access steps described under Data Sources, you may download the datasets directly or via the command line. The example below mirrors the dialog shown in the PhysioNet file browser.

```
# Replace USERNAME with your PhysioNet username.
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimiciv/3.1/
```

Extraction in this repository is written against the BigQuery copies of both datasets. Before the first run, set `BILLING_PROJECT` in `utils/extraction.py` to your own Google Cloud project and authenticate:

```
gcloud auth application-default login
```

## 🧬 Data Sources

The framework was developed and evaluated using data derived from two critical care databases. Both require credentialed access, and neither the data nor any derived patient-level file is redistributed in this repository.

- **MIMIC-IV v3.1**
  Access: https://physionet.org/content/mimiciv/3.1/ (credentialed access required).
- **eICU-CRD v2.0**
  Access: https://physionet.org/content/eicu-crd/2.0/ (credentialed access required).

## 🔁 Reproducibility

Run `DINIRS.ipynb` from top to bottom. The workflow proceeds through the following stages:

1. **Configuration**: random seed (42), architecture, loss weights, and cross-fitting settings.
2. **Cohort extraction**: MIMIC-IV cohort, VFD-28 outcome, baseline covariates, and temporal tensors.
3. **Preprocessing**: scaling, padding masks, and the cross-fitted tree base.
4. **Training and cross-fitting**: two-stage training over 5 folds to produce out-of-fold ITEs.
5. **Baselines and ensemble**: T-Learner, causal forest, and causal survival forest comparisons.
6. **Clinical impact**: policy value, subgroup analyses, and multiple imputation with Rubin's rules.
7. **External validation**: the MIMIC-trained fold models applied unchanged to eICU-CRD.
8. **Figures and export**: all figures reported in the manuscript.

Extraction results will be cached under `output/` on first run. Subsequent runs re-use the cache. Please set `FORCE_EXTRACT_MIMIC`, `FORCE_EXTRACT_EICU`, or `FORCE_RETRAIN` to `True` in the configuration cell to regenerate from source.

## Citation

If you use this work, please cite: 

Islam MF, Mosier J, Subbian V. DINIRS: Digital Twin for Individualized Treatment Effects of Non-Invasive Respiratory Support Strategies. *Under review*.


## 📌 Dependencies

This repository is implemented using Python 3.13 and requires the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

A CUDA-capable GPU is optional; the notebook falls back to CPU automatically.

## 📬 Contact

For questions or collaboration inquiries, please contact:

Md Fantacher Islam
PhD Student, Systems and Industrial Engineering
University of Arizona
Email: [fantacher@arizona.edu]
