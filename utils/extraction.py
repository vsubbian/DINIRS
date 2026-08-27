"""MIMIC-IV and eICU-CRD extraction, dataset construction, and matching."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os, time, warnings
from sklearn.neighbors import NearestNeighbors


COVARIATES_23 = [
    'age',
    'gender',
    'bmi',

    'heart_rate',
    'resp_rate',
    'spo2',
    'mbp',
    'temperature',
    'fio2',
    'peep',

    'pao2',
    'paco2',
    'ph',
    'pf_ratio',

    'lactate',
    'creatinine',
    'bilirubin',
    'platelets',
    'wbc',

    'sofa_score',
    'gcs_total',

    'hours_since_icu_admit',
    'vasopressor_flag',
]

MIMIC_ITEMIDS = {
    'heart_rate': [220045],
    'resp_rate': [220210, 224690],
    'spo2': [220277],
    'mbp': [220052, 220181, 225312],
    'temperature': [223761, 223762],
    'fio2': [223835],
    'peep': [220339, 224700],
    'pao2': [220224],
    'paco2': [220235],
    'ph': [220274, 220734],
    'lactate': [225668],
    'creatinine': [220615],
    'bilirubin': [225690],
    'platelets': [227457],
    'wbc': [220546],
    'gcs_total': [220739, 223900, 223901],
}


def compute_vfd28(vent_duration_days, survived_28d):
    """Compute Ventilator-Free Days at 28 days (VFD-28)."""
    vfd28 = np.where(
        survived_28d == 1,
        np.maximum(0, 28 - vent_duration_days),
        0.0
    )
    return vfd28.astype(np.float32), survived_28d.astype(np.float32)


class NIRSTwinDataset(Dataset):
    """PyTorch Dataset for DINIRS training."""

    def __init__(self, sequences, treatments, vfd_observed, delta, pad_masks=None,
                 tau_base=None, ipcw=None):
        self.sequences = torch.FloatTensor(sequences)
        self.treatments = torch.FloatTensor(treatments).unsqueeze(-1)
        self.vfd_observed = torch.FloatTensor(vfd_observed).unsqueeze(-1)
        self.delta = torch.FloatTensor(delta).unsqueeze(-1)

        self.tau_base = (torch.FloatTensor(np.asarray(tau_base)).unsqueeze(-1)
                         if tau_base is not None else None)
        self.ipcw = (torch.FloatTensor(np.asarray(ipcw)).unsqueeze(-1)
                     if ipcw is not None else None)

        if pad_masks is not None:
            self.pad_masks = torch.BoolTensor(pad_masks)
        else:
            self.pad_masks = torch.zeros(len(sequences), sequences.shape[1],
                                         dtype=torch.bool)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            'x': self.sequences[idx],
            'treatment': self.treatments[idx],
            'vfd': self.vfd_observed[idx],
            'delta': self.delta[idx],
            'pad_mask': self.pad_masks[idx],
        }
        if self.tau_base is not None:
            item['tau_base'] = self.tau_base[idx]
        if self.ipcw is not None:
            item['ipcw'] = self.ipcw[idx]
        return item


def propensity_score_matching(df, treatment_col, covariate_cols,
                               caliper=0.05, random_state=42):
    """Propensity Score Matching with caliper."""
    X = df[covariate_cols].values
    W = df[treatment_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
    ps_model.fit(X_scaled, W)

    ps = ps_model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df['propensity_score'] = ps

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    matched_indices = []
    used_controls = set()

    for idx, row in treated.iterrows():
        ps_treat = row['propensity_score']
        candidates = control[~control.index.isin(used_controls)]

        if len(candidates) == 0:
            continue

        distances = np.abs(candidates['propensity_score'] - ps_treat)
        best_idx = distances.idxmin()
        best_dist = distances.min()

        if best_dist <= caliper:
            matched_indices.append(idx)
            matched_indices.append(best_idx)
            used_controls.add(best_idx)

    df_matched = df.loc[matched_indices].copy()

    return df_matched, ps_model


def create_dataloaders(
    sequences,
    treatments,
    vfd_observed,
    delta,
    pad_masks=None,
    batch_size=128,
    val_fraction=0.10,
    test_fraction=0.10,
    random_state=42,
    return_indices=False,
    tau_base=None,
):
    """Create train/val/test DataLoaders."""
    N = len(sequences)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(N)

    n_test = int(N * test_fraction)
    n_val = int(N * val_fraction)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    def make_loader(idx, shuffle):
        ds = NIRSTwinDataset(
            sequences=sequences[idx],
            treatments=treatments[idx],
            vfd_observed=vfd_observed[idx],
            delta=delta[idx],
            pad_masks=pad_masks[idx] if pad_masks is not None else None,
            tau_base=tau_base[idx] if tau_base is not None else None,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    if return_indices:
        split_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        return train_loader, val_loader, test_loader, split_idx

    return train_loader, val_loader, test_loader

loader_compute_vfd28 = compute_vfd28


warnings.filterwarnings("ignore", category=UserWarning, module='google.cloud.bigquery')
warnings.filterwarnings("ignore", category=FutureWarning)


BILLING_PROJECT = "nirs-484505"
DATA_PROJECT    = "physionet-data"

MIMIC = {
    "icu":     "physionet-data.mimiciv_3_1_icu",
    "hosp":    "physionet-data.mimiciv_3_1_hosp",
    "derived": "physionet-data.mimiciv_3_1_derived",
}
EICU = {
    "main":    "physionet-data.eicu_crd",
    "derived": "physionet-data.eicu_crd_derived",
}

ITEMIDS = {
    "inv_proc": [225792],
    "niv_proc": [225794, 225949, 227578],
    "hfnc":     [226732],
    "bipap":    [227578, 227579],
    "cpap":     [227580],
    "peep":     [220339],
    "fio2":     [223835, 223834],
    "tv":       [224685, 224684],
}

VENT_MAP = {
    'InvasiveVent':       'Invasive',
    'Tracheostomy':       'Invasive',
    'NonInvasiveVent':    'NIV',
    'HFNC':               'NIV',
    'SupplementalOxygen': 'Oxygen',
    'None':               'None',
}
VENT_STATUS_NIRS = {"NonInvasiveVent", "HFNC"}
VENT_STATUS_IMV  = {"InvasiveVent", "Tracheostomy"}

FEATURE_COLS = [
    "age_X", "gender_X", "bmi_X",
    "sofa_X", "gcs_X", "sapsii_X",
    "hr_mean_X", "rr_mean_X", "spo2_mean_X", "mbp_mean_X", "tempc_mean_X",
    "pao2_X", "paco2_X", "ph_X", "fio2_X", "lactate_X", "bicarbonate_X",
    "pf_ratio_X", "rox_index_X",
    "copd_X", "chf_X", "immunosuppressed_X", "sepsis_X",
]

CONTINUOUS_COLS = [c for c in FEATURE_COLS if c not in
                   {"gender_X", "copd_X", "chf_X", "immunosuppressed_X", "sepsis_X"}]
BINARY_COLS = ["gender_X", "copd_X", "chf_X", "immunosuppressed_X", "sepsis_X"]

T0_WINDOW_H      = 24
MIN_LOS_DAYS     = 0.5
VFD_HORIZON_DAYS = 28

SUPPORT_WINDOW_HOURS = 24.0

EXCLUDE_DIED_24H = True

MAX_MISSING_COVARIATE_FRAC = 0.15
APPLY_MISSING_COVARIATE_FILTER = False

PMM_DONORS = 5

LAST_BASELINE_RAW = None
RANDOM_STATE     = 42


_client = None

def init_client():
    """Initialize BigQuery client (billing to your project)."""
    global _client
    from google.cloud import bigquery
    os.environ["GOOGLE_CLOUD_PROJECT"] = BILLING_PROJECT
    _client = bigquery.Client(project=BILLING_PROJECT)
    return _client


def run_bq(sql, verbose=True):
    """Execute BigQuery SQL and return DataFrame (from m0_config.py)."""
    global _client
    if _client is None:
        init_client()
    t0 = time.time()
    df = _client.query(sql).to_dataframe()
    return df


def ids_str(id_array):
    """Convert array of IDs to comma-separated string for SQL IN clause."""
    return ",".join(str(int(x)) for x in id_array)


def safe_divide(a, b, fill=np.nan):
    """Element-wise division, filling division-by-zero with fill."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


def extract_icu_stays():
    """Pull adult ICU stays with LOS >= 0.5 days."""
    sql = f"""
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime   AS icu_intime,
        ie.outtime  AS icu_outtime,
        DATETIME_DIFF(ie.outtime, ie.intime, HOUR) / 24.0 AS los_icu_days,
        pat.anchor_age AS age,
        pat.gender,
        adm.deathtime,
        adm.hospital_expire_flag,
        adm.admission_type
    FROM `{MIMIC['icu']}.icustays` ie
    JOIN `{MIMIC['hosp']}.patients`   pat ON ie.subject_id = pat.subject_id
    JOIN `{MIMIC['hosp']}.admissions` adm ON ie.hadm_id    = adm.hadm_id
    WHERE pat.anchor_age >= 18
      AND DATETIME_DIFF(ie.outtime, ie.intime, HOUR) / 24.0 >= {MIN_LOS_DAYS}
    """
    df = run_bq(sql)
    return df


def identify_arf(df_stays):
    """ARF = ICD-10 J96.0x PLUS physiological confirmation."""
    sql_icd_broad = f"""
    SELECT DISTINCT d.hadm_id
    FROM `{MIMIC['hosp']}.diagnoses_icd` d
    WHERE d.icd_version = 10
      AND (d.icd_code LIKE 'J960%'
           OR d.icd_code LIKE 'J80%'
           OR d.icd_code LIKE 'J96%')
    """
    df_icd = run_bq(sql_icd_broad)

    sql_physio = f"""
    WITH physio_flags AS (
        SELECT
            ie.stay_id,
            ie.hadm_id,
            CASE WHEN v.spo2_min < 94 THEN 1 ELSE 0 END AS flag_spo2,
            CASE WHEN v.resp_rate_max > 25 THEN 1 ELSE 0 END AS flag_rr,
            CASE WHEN bg_agg.pao2_min < 60 THEN 1 ELSE 0 END AS flag_pao2,
            CASE WHEN bg_agg.paco2_max > 50 AND bg_agg.ph_min < 7.35
                 THEN 1 ELSE 0 END AS flag_hypercapnic
        FROM `{MIMIC['icu']}.icustays` ie
        JOIN `{MIMIC['hosp']}.patients` pat
            ON ie.subject_id = pat.subject_id
        LEFT JOIN `{MIMIC['derived']}.first_day_vitalsign` v
            ON ie.stay_id = v.stay_id
        LEFT JOIN (
            SELECT
                i2.stay_id,
                MIN(b.po2)     AS pao2_min,
                MAX(b.pco2)    AS paco2_max,
                MIN(b.ph)      AS ph_min
            FROM `{MIMIC['derived']}.bg` b
            JOIN `{MIMIC['derived']}.icustay_detail` i2
                ON b.subject_id = i2.subject_id
            WHERE b.charttime BETWEEN i2.icu_intime
                  AND DATETIME_ADD(i2.icu_intime, INTERVAL 24 HOUR)
              AND i2.los_icu >= {MIN_LOS_DAYS}
            GROUP BY i2.stay_id
        ) bg_agg ON ie.stay_id = bg_agg.stay_id
        WHERE pat.anchor_age >= 18
          AND DATETIME_DIFF(ie.outtime, ie.intime, HOUR) / 24.0 >= {MIN_LOS_DAYS}
    )
    SELECT stay_id, hadm_id,
           flag_spo2, flag_rr, flag_pao2, flag_hypercapnic,
           GREATEST(flag_spo2, flag_rr, flag_pao2, flag_hypercapnic) AS any_physio
    FROM physio_flags
    """
    df_physio = run_bq(sql_physio)

    arf_hadm = set(df_icd["hadm_id"])
    df_physio_confirmed = df_physio[
        (df_physio["hadm_id"].isin(arf_hadm)) &
        (df_physio["any_physio"] == 1)
    ]
    arf_stay_ids = set(df_physio_confirmed["stay_id"])
    df_arf = df_stays[df_stays["stay_id"].isin(arf_stay_ids)].copy()
    return df_arf


def apply_exclusions(df_arf):
    """Exclude DNR/DNI, crash intubation, chronic vent."""
    stay_ids = ids_str(df_arf["stay_id"])
    n0 = len(df_arf)

    sql_dnr = f"""
    SELECT DISTINCT stay_id, value
    FROM `{MIMIC['icu']}.chartevents`
    WHERE itemid = 223758
      AND value IS NOT NULL
      AND REGEXP_CONTAINS(UPPER(value),
            r'DNR|DNI|DO NOT RESUSCITATE|DO NOT INTUBATE|COMFORT MEASURES|CMO')
      AND stay_id IN ({stay_ids})
    """
    df_dnr = run_bq(sql_dnr)
    excl_dnr = set(df_dnr["stay_id"])

    sql_crash = f"""
    SELECT DISTINCT pe.stay_id
    FROM `{MIMIC['icu']}.procedureevents` pe
    JOIN `{MIMIC['icu']}.icustays` ie ON pe.stay_id = ie.stay_id
    WHERE pe.itemid = 225792
      AND pe.stay_id IN ({stay_ids})
      AND DATETIME_DIFF(pe.starttime, ie.intime, MINUTE) <= 60
    """
    df_crash = run_bq(sql_crash)
    excl_crash = set(df_crash["stay_id"])

    sql_chronic = f"""
    SELECT DISTINCT d.hadm_id
    FROM `{MIMIC['hosp']}.diagnoses_icd` d
    WHERE d.icd_version = 10
      AND (d.icd_code LIKE 'Z991%'
           OR d.icd_code LIKE 'J950%'
           OR d.icd_code LIKE 'Z930%')
    """
    df_chronic = run_bq(sql_chronic)
    excl_chronic_hadm = set(df_chronic["hadm_id"])
    excl_chronic = set(df_arf[df_arf["hadm_id"].isin(excl_chronic_hadm)]["stay_id"])

    sql_died24 = f"""
    SELECT DISTINCT ie.stay_id
    FROM `{MIMIC['icu']}.icustays` ie
    JOIN `{MIMIC['hosp']}.admissions` adm ON ie.hadm_id = adm.hadm_id
    WHERE ie.stay_id IN ({stay_ids})
      AND adm.deathtime IS NOT NULL
      AND DATETIME_DIFF(adm.deathtime, ie.intime, HOUR) <= 24
    """
    if EXCLUDE_DIED_24H:
        df_died24 = run_bq(sql_died24)
        excl_died24 = set(df_died24["stay_id"])
    else:
        excl_died24 = set()

    all_excl = excl_dnr | excl_crash | excl_chronic | excl_died24
    df_clean = df_arf[~df_arf["stay_id"].isin(all_excl)].copy()

    return df_clean


def assign_treatment(df_cohort):
    """Treatment assignment with ventilation correction from procedureevents."""
    stay_ids = ids_str(df_cohort["stay_id"])

    sql_vent = f"""
    SELECT v.stay_id, v.starttime, v.endtime, v.ventilation_status
    FROM `{MIMIC['derived']}.ventilation` v
    WHERE v.stay_id IN ({stay_ids})
    ORDER BY v.stay_id, v.starttime
    """

    niv_ids = ",".join(map(str, ITEMIDS["niv_proc"]))
    inv_ids = ",".join(map(str, ITEMIDS["inv_proc"]))
    sql_proc = f"""
    SELECT pe.stay_id, pe.starttime, pe.endtime, pe.itemid,
        CASE WHEN pe.itemid = 225792 THEN 'Invasive' ELSE 'NIV' END AS vent_type
    FROM `{MIMIC['icu']}.procedureevents` pe
    WHERE pe.stay_id IN ({stay_ids})
      AND pe.itemid IN ({niv_ids},{inv_ids})
    ORDER BY pe.stay_id, pe.starttime
    """

    df_vent = run_bq(sql_vent)
    df_proc = run_bq(sql_proc)

    priority_map = {'Invasive': 1, 'NIV': 2}

    df_merged = pd.merge(
        df_vent, df_proc,
        on='stay_id', suffixes=('_status', '_proc')
    )

    df_overlaps = df_merged.query(
        '(starttime_status <= endtime_proc and endtime_status >= starttime_proc) and '
        'ventilation_status in ["None", "SupplementalOxygen"]'
    ).copy()

    if len(df_overlaps) > 0:
        df_overlaps['priority'] = df_overlaps['vent_type'].map(priority_map)
        corrections = df_overlaps.groupby(
            ['stay_id', 'starttime_status', 'endtime_status', 'ventilation_status']
        )['priority'].min()
        corrections_map = {1: 'InvasiveVent', 2: 'NonInvasiveVent'}
        corrections = corrections.map(corrections_map).reset_index()
        corrections = corrections.rename(columns={
            'starttime_status': 'starttime',
            'endtime_status':   'endtime',
            'priority':         'corrected_status',
        })

        df_vent_corrected = pd.merge(
            df_vent, corrections,
            on=['stay_id', 'starttime', 'endtime', 'ventilation_status'],
            how='left'
        )
        df_vent_corrected['final_status'] = np.where(
            pd.notna(df_vent_corrected['corrected_status']),
            df_vent_corrected['corrected_status'],
            df_vent_corrected['ventilation_status']
        )
    else:
        df_vent_corrected = df_vent.copy()
        df_vent_corrected['final_status'] = df_vent_corrected['ventilation_status']

    df_vent_corrected['vent_type'] = df_vent_corrected['final_status'].map(VENT_MAP)
    df_vent_corrected = df_vent_corrected.sort_values(['stay_id', 'starttime'])

    n_before = df_vent[df_vent['ventilation_status'].isin(
        ['NonInvasiveVent', 'HFNC'])].shape[0]
    n_after = df_vent_corrected[df_vent_corrected['vent_type'] == 'NIV'].shape[0]

    df_vent_events = df_vent_corrected[
        df_vent_corrected['vent_type'].isin(['Invasive', 'NIV'])
    ].copy()

    df_first_events = (
        df_vent_events
        .groupby(['stay_id', 'vent_type'])['starttime']
        .min()
        .unstack()
    )

    all_stay_ids = df_cohort['stay_id'].unique()
    df_first_events = df_first_events.reindex(all_stay_ids)

    def _categorize(row):
        inv_exists = pd.notna(row.get('Invasive'))
        niv_exists = pd.notna(row.get('NIV'))
        if inv_exists and not niv_exists:
            return 'IMV_only', 0
        if niv_exists and not inv_exists:
            return 'NIRS_only', 1
        if not inv_exists and not niv_exists:
            return 'None', np.nan
        if row['Invasive'] < row['NIV']:
            return 'IMV_then_NIRS', 0
        if row['NIV'] < row['Invasive']:
            return 'NIRS_then_IMV', 1
        return 'ambiguous', np.nan

    results = df_first_events.apply(_categorize, axis=1)
    df_tx = pd.DataFrame({
        'stay_id': df_first_events.index,
        'category': [r[0] for r in results],
        'Treatment_W': [r[1] for r in results],
    })

    _first = df_first_events.min(axis=1)
    _intime = (df_cohort.drop_duplicates('stay_id')
                        .set_index('stay_id')['icu_intime'])
    _hrs = ((pd.to_datetime(_first, errors='coerce')
             - pd.to_datetime(_intime, errors='coerce'))
            .dt.total_seconds() / 3600.0)
    df_tx['hrs_to_support'] = df_tx['stay_id'].map(_hrs)
    _drop = ~(df_tx['hrs_to_support'] <= SUPPORT_WINDOW_HOURS)
    df_tx = df_tx[~_drop].copy()

    df_tx = df_tx[df_tx['Treatment_W'].notna()].copy()
    df_tx['Treatment_W'] = df_tx['Treatment_W'].astype(int)

    df_treat = df_tx[['stay_id', 'Treatment_W', 'category']]
    df_final = df_cohort.merge(df_treat, on='stay_id', how='inner')

    return df_final


def build_cohort():
    """End-to-end cohort construction."""
    df_stays = extract_icu_stays()
    df_arf   = identify_arf(df_stays)
    df_clean = apply_exclusions(df_arf)
    df_final = assign_treatment(df_clean)
    return df_final


def compute_vfd28(df_cohort):
    """VFD-28 from ventilation timeline clipped to 28-day window."""
    stay_ids = ids_str(df_cohort["stay_id"])

    sql_vent_full = f"""
    WITH vent_with_icu AS (
        SELECT
            v.stay_id, v.starttime, v.endtime, v.ventilation_status,
            ie.intime AS icu_intime,
            GREATEST(v.starttime, ie.intime) AS eff_start,
            LEAST(v.endtime, DATETIME_ADD(ie.intime, INTERVAL 28 DAY)) AS eff_end
        FROM `{MIMIC['derived']}.ventilation` v
        JOIN `{MIMIC['icu']}.icustays` ie ON v.stay_id = ie.stay_id
        WHERE v.stay_id IN ({stay_ids})
          AND v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
          AND v.starttime < DATETIME_ADD(ie.intime, INTERVAL 28 DAY)
          AND v.endtime > ie.intime
    )
    SELECT stay_id,
           SUM(DATETIME_DIFF(eff_end, eff_start, HOUR)) AS total_imv_hours
    FROM vent_with_icu
    WHERE eff_end > eff_start
    GROUP BY stay_id
    """
    df_imv = run_bq(sql_vent_full)

    sql_death = f"""
    SELECT
        ie.stay_id,
        adm.deathtime,
        adm.hospital_expire_flag,
        CASE
            WHEN adm.deathtime IS NOT NULL
                 AND DATETIME_DIFF(adm.deathtime, ie.intime, DAY) <= {VFD_HORIZON_DAYS}
            THEN 1
            WHEN adm.deathtime IS NULL AND adm.hospital_expire_flag = 1
            THEN 1
            ELSE 0
        END AS died_28d,
        CASE
            WHEN adm.deathtime IS NOT NULL
            THEN DATETIME_DIFF(adm.deathtime, ie.intime, DAY)
            ELSE NULL
        END AS days_to_death
    FROM `{MIMIC['icu']}.icustays` ie
    JOIN `{MIMIC['hosp']}.admissions` adm ON ie.hadm_id = adm.hadm_id
    WHERE ie.stay_id IN ({stay_ids})
    """
    df_death = run_bq(sql_death)

    df_out = df_cohort[["stay_id", "Treatment_W"]].copy()
    df_out = df_out.merge(df_imv, on="stay_id", how="left")
    df_out = df_out.merge(
        df_death[["stay_id", "died_28d", "days_to_death"]],
        on="stay_id", how="left"
    )

    df_out["total_imv_hours"] = df_out["total_imv_hours"].fillna(0)
    df_out["died_28d"] = df_out["died_28d"].fillna(0).astype(int)
    df_out["total_imv_days"] = df_out["total_imv_hours"] / 24.0
    df_out["vfd28"] = np.where(
        df_out["died_28d"] == 1,
        0.0,
        np.clip(28.0 - df_out["total_imv_days"], 0, 28)
    )
    df_out["delta"] = 1 - df_out["died_28d"]

    for w in [0, 1]:
        arm = df_out[df_out["Treatment_W"] == w]
        lbl = "NIRS" if w == 1 else "IMV"
    return df_out


def extract_baseline_covariates(df_cohort):
    """Extract 23 baseline (first-24h) covariates."""
    stay_ids = ids_str(df_cohort["stay_id"])

    sql_base = f"""
    SELECT
        ie.stay_id,
        pat.anchor_age AS age_X,
        CASE WHEN pat.gender = 'M' THEN 1 ELSE 0 END AS gender_X,
        fdw.weight_admit,
        fdh.height,
        CASE
            WHEN fdh.height > 0 AND fdw.weight_admit > 0
            THEN fdw.weight_admit / POWER(fdh.height / 100.0, 2)
            ELSE NULL
        END AS bmi_X,
        sf.sofa AS sofa_X,
        gcs.gcs_min AS gcs_X,
        v.heart_rate_mean AS hr_mean_X,
        v.resp_rate_mean AS rr_mean_X,
        v.spo2_mean AS spo2_mean_X,
        v.mbp_mean AS mbp_mean_X,
        v.temperature_mean AS tempc_mean_X
    FROM `{MIMIC['icu']}.icustays` ie
    JOIN `{MIMIC['hosp']}.patients` pat ON ie.subject_id = pat.subject_id
    LEFT JOIN `{MIMIC['derived']}.first_day_height` fdh ON ie.stay_id = fdh.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_weight` fdw ON ie.stay_id = fdw.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_sofa` sf ON ie.stay_id = sf.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_gcs` gcs ON ie.stay_id = gcs.stay_id
    LEFT JOIN (
        SELECT stay_id, MAX(sapsii) AS sapsii
        FROM `{MIMIC['derived']}.sapsii`
        GROUP BY stay_id
    ) sp ON ie.stay_id = sp.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_vitalsign` v ON ie.stay_id = v.stay_id
    WHERE ie.stay_id IN ({stay_ids})
    """
    sql_base = sql_base.replace(
        "v.temperature_mean AS tempc_mean_X",
        "v.temperature_mean AS tempc_mean_X,\n        sp.sapsii AS sapsii_X"
    )

    df_base = run_bq(sql_base)
    df_base.drop(columns=["weight_admit", "height"], inplace=True, errors="ignore")

    W_BG  = "b.charttime  BETWEEN i.icu_intime AND DATETIME_ADD(i.icu_intime, INTERVAL 24 HOUR)"
    W_LAB = "le.charttime BETWEEN i.icu_intime AND DATETIME_ADD(i.icu_intime, INTERVAL 24 HOUR)"
    W_CE  = "ce.charttime BETWEEN ie.intime    AND DATETIME_ADD(ie.intime,    INTERVAL 24 HOUR)"
    JBG  = (f"FROM `{MIMIC['derived']}.bg` b "
            f"JOIN `{MIMIC['derived']}.icustay_detail` i ON b.subject_id=i.subject_id "
            f"WHERE i.stay_id IN ({stay_ids}) AND {W_BG}")
    JLAB = (f"FROM `{MIMIC['hosp']}.labevents` le "
            f"JOIN `{MIMIC['derived']}.icustay_detail` i ON le.subject_id=i.subject_id "
            f"WHERE i.stay_id IN ({stay_ids}) AND {W_LAB}")
    JCE  = (f"FROM `{MIMIC['icu']}.chartevents` ce "
            f"JOIN `{MIMIC['icu']}.icustays` ie ON ce.stay_id=ie.stay_id "
            f"WHERE ce.stay_id IN ({stay_ids}) AND {W_CE}")
    parts = [
        f"SELECT i.stay_id, 'po2' AS var, b.po2 AS val {JBG} AND b.po2 IS NOT NULL",
        f"SELECT i.stay_id, 'pco2', b.pco2 {JBG} AND b.pco2 IS NOT NULL",
        f"SELECT i.stay_id, 'ph', b.ph {JBG} AND b.ph IS NOT NULL",
        f"SELECT i.stay_id, 'lactate', b.lactate {JBG} AND b.lactate IS NOT NULL",
        f"SELECT i.stay_id, 'bicarbonate', b.bicarbonate {JBG} AND b.bicarbonate IS NOT NULL",
        f"SELECT i.stay_id, 'fio2', CASE WHEN b.fio2<=1.0 THEN b.fio2*100 ELSE b.fio2 END {JBG} AND b.fio2 IS NOT NULL",
        f"SELECT i.stay_id, 'bicarbonate', le.valuenum {JLAB} AND le.itemid=50882 AND le.valuenum BETWEEN 5 AND 60",
        f"SELECT i.stay_id, 'lactate', le.valuenum {JLAB} AND le.itemid=50813 AND le.valuenum BETWEEN 0.1 AND 30",
        f"SELECT i.stay_id, 'po2', le.valuenum {JLAB} AND le.itemid=50821 AND le.valuenum BETWEEN 20 AND 600",
        f"SELECT i.stay_id, 'pco2', le.valuenum {JLAB} AND le.itemid=50818 AND le.valuenum BETWEEN 10 AND 150",
        f"SELECT i.stay_id, 'ph', le.valuenum {JLAB} AND le.itemid=50820 AND le.valuenum BETWEEN 6.8 AND 7.8",
        f"SELECT i.stay_id, 'fio2', le.valuenum {JLAB} AND le.itemid=50816 AND le.valuenum BETWEEN 21 AND 100",
        f"SELECT ce.stay_id, 'fio2', CASE WHEN ce.valuenum<=1.0 THEN ce.valuenum*100 ELSE ce.valuenum END {JCE} AND ce.itemid=223835 AND ce.valuenum IS NOT NULL",
        f"SELECT ce.stay_id, 'fio2', LEAST(21+3*ce.valuenum,100) {JCE} AND ce.itemid=223834 AND ce.valuenum BETWEEN 0 AND 60",
    ]
    sql_abg = (
        "WITH gas_long AS (\n  " + "\n  UNION ALL ".join(parts) + "\n)\n"
        "SELECT stay_id,"
        " AVG(IF(var='po2',val,NULL))         AS pao2_X,"
        " AVG(IF(var='pco2',val,NULL))        AS paco2_X,"
        " AVG(IF(var='ph',val,NULL))          AS ph_X,"
        " AVG(IF(var='fio2',val,NULL))        AS fio2_X,"
        " AVG(IF(var='lactate',val,NULL))     AS lactate_X,"
        " AVG(IF(var='bicarbonate',val,NULL)) AS bicarbonate_X"
        " FROM gas_long GROUP BY stay_id"
    )
    df_abg = run_bq(sql_abg)

    hadm_ids = ids_str(df_cohort["hadm_id"])
    sql_charlson = f"""
    SELECT
        ie.stay_id,
        COALESCE(c.chronic_pulmonary_disease, 0) AS copd_X,
        COALESCE(c.congestive_heart_failure, 0)  AS chf_X,
        CASE
            WHEN COALESCE(c.aids, 0) = 1
              OR COALESCE(c.metastatic_solid_tumor, 0) = 1
            THEN 1 ELSE 0
        END AS immunosuppressed_X
    FROM `{MIMIC['icu']}.icustays` ie
    LEFT JOIN `{MIMIC['derived']}.charlson` c ON ie.hadm_id = c.hadm_id
    WHERE ie.stay_id IN ({stay_ids})
    """
    sql_sepsis = f"""
    SELECT DISTINCT stay_id, 1 AS sepsis_X
    FROM `{MIMIC['derived']}.sepsis3`
    WHERE stay_id IN ({stay_ids})
    """
    df_charlson = run_bq(sql_charlson)
    df_sepsis = run_bq(sql_sepsis)

    df = df_base.merge(df_abg, on="stay_id", how="left")
    df = df.merge(df_charlson, on="stay_id", how="left")
    df = df.merge(df_sepsis, on="stay_id", how="left")
    df["sepsis_X"] = df["sepsis_X"].fillna(0).astype(int)

    df["bmi_X"] = np.clip(df["bmi_X"], 14.0, 70.0)

    _cont = [c for c in CONTINUOUS_COLS if c in df.columns]
    if _cont and APPLY_MISSING_COVARIATE_FILTER:
        _miss_frac = df[_cont].isna().mean(axis=1)
        df = df[_miss_frac <= MAX_MISSING_COVARIATE_FRAC].copy()

    global LAST_BASELINE_RAW
    _raw = df.copy()
    for _bc in BINARY_COLS:
        if _bc in _raw.columns:
            _raw[_bc] = _raw[_bc].fillna(0).astype(int)
    _raw["pf_ratio_X"] = np.clip(safe_divide(
        _raw["pao2_X"].values, _raw["fio2_X"].values / 100.0), 0, 700)
    _raw["rox_index_X"] = np.clip(safe_divide(
        safe_divide(_raw["spo2_mean_X"].values, _raw["fio2_X"].values / 100.0),
        _raw["rr_mean_X"].values), 0, 30)
    LAST_BASELINE_RAW = _raw

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    _rng_imp = np.random.RandomState(42)
    _pmm_pred = [c for c in CONTINUOUS_COLS if c in df.columns]
    for col in CONTINUOUS_COLS:
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        if n_miss == 0:
            continue
        obs = df[col].notna().values
        med = float(df[col].median())
        preds = [c for c in _pmm_pred if c != col]
        if not preds or obs.sum() < PMM_DONORS * 2:
            df[col] = df[col].fillna(med)
            continue
        Z = df[preds].astype(float)
        Z = Z.fillna(Z.median()).values
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-9, 1.0, Z.std(0))
        Z = np.column_stack([np.ones(len(Z)), Z])
        y = df[col].astype(float).values
        A = Z[obs]
        beta = np.linalg.solve(A.T @ A + 1e-3 * np.eye(Z.shape[1]), A.T @ y[obs])
        yhat = Z @ beta
        donor_pred, donor_val = yhat[obs], y[obs]
        order = np.argsort(donor_pred)
        dp, dv = donor_pred[order], donor_val[order]
        miss_idx = np.where(~obs)[0]
        pos = np.searchsorted(dp, yhat[miss_idx])
        drawn = np.empty(len(miss_idx))
        for t, (m, pp) in enumerate(zip(miss_idx, pos)):
            lo = max(0, pp - PMM_DONORS)
            hi = min(len(dv), pp + PMM_DONORS)
            cand = np.argsort(np.abs(dp[lo:hi] - yhat[m]))[:PMM_DONORS] + lo
            drawn[t] = dv[_rng_imp.choice(cand)]
        df.loc[df.index[miss_idx], col] = drawn

    df["pf_ratio_X"] = safe_divide(
        df["pao2_X"].values, df["fio2_X"].values / 100.0)
    df["pf_ratio_X"] = np.clip(df["pf_ratio_X"], 0, 700)

    df["rox_index_X"] = safe_divide(
        safe_divide(df["spo2_mean_X"].values, df["fio2_X"].values / 100.0),
        df["rr_mean_X"].values)
    df["rox_index_X"] = np.clip(df["rox_index_X"], 0, 30)

    present = [c for c in FEATURE_COLS if c in df.columns]
    return df


def standardize_features(df, cols=None, reference_df=None):
    """Z-score standardization for continuous features."""
    if cols is None:
        cols = CONTINUOUS_COLS
    ref = (reference_df if reference_df is not None else df).copy()
    stats = {}
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors='coerce')
    for col in cols:
        if col in df_out.columns:
            mu  = float(ref[col].mean())
            sig = float(ref[col].std())
            if sig < 1e-10:
                sig = 1.0
            df_out[col] = (df_out[col] - mu) / sig
            stats[col] = (mu, sig)
    return df_out, stats


def extract_temporal_chartevents(df_cohort, chunk_size=5000):
    """Extract time-series covariates from chartevents for Transformer input."""
    stay_ids = df_cohort['stay_id'].tolist()
    chunks = [stay_ids[i:i + chunk_size]
              for i in range(0, len(stay_ids), chunk_size)]

    all_vital_ids = []
    for ids in MIMIC_ITEMIDS.values():
        all_vital_ids.extend(ids)
    vital_ids_str = ','.join(map(str, all_vital_ids))

    results = []
    for i, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        query = f"""
        SELECT
            ce.stay_id,
            ce.charttime,
            AVG(IF(itemid = 220045, valuenum, NULL))                         AS heart_rate,
            AVG(IF(itemid IN (220210, 224690), valuenum, NULL))              AS resp_rate,
            AVG(IF(itemid = 220277 AND valuenum <= 100, valuenum, NULL))     AS spo2,
            AVG(IF(itemid IN (220052, 220181, 225312), valuenum, NULL))      AS mbp,
            AVG(CASE
                    WHEN itemid = 223761 AND valuenum BETWEEN 70 AND 120
                         THEN (valuenum - 32) * 5.0/9.0
                    WHEN itemid = 223762 AND valuenum BETWEEN 10 AND 50
                         THEN valuenum
                    ELSE NULL END)                                           AS temperature,
            AVG(CASE
                    WHEN itemid = 223835 AND valuenum <= 1.0 THEN valuenum * 100
                    WHEN itemid = 223835 AND valuenum >  1.0 THEN valuenum
                    WHEN itemid = 223834 AND valuenum BETWEEN 0 AND 60 THEN LEAST(21 + 3*valuenum, 100)
                    ELSE NULL END)                                           AS fio2,
            AVG(IF(itemid IN (220339, 224700), valuenum, NULL))              AS peep,
            AVG(IF(itemid = 220224, valuenum, NULL))                         AS pao2,
            AVG(IF(itemid = 220235, valuenum, NULL))                         AS paco2,
            AVG(IF(itemid IN (220274, 220734), valuenum, NULL))              AS ph,
            AVG(IF(itemid = 225668, valuenum, NULL))                         AS lactate,
            AVG(IF(itemid = 220615, valuenum, NULL))                         AS creatinine,
            AVG(IF(itemid = 225690, valuenum, NULL))                         AS bilirubin,
            AVG(IF(itemid = 227457, valuenum, NULL))                         AS platelets,
            AVG(IF(itemid = 220546, valuenum, NULL))                         AS wbc,
            AVG(IF(itemid IN (220739, 223900, 223901), valuenum, NULL))      AS gcs_component
        FROM `{MIMIC['icu']}.chartevents` ce
        JOIN `{MIMIC['icu']}.icustays` ie2 ON ce.stay_id = ie2.stay_id
        WHERE ce.stay_id IN ({chunk_str})
          AND ce.itemid IN ({vital_ids_str})
          AND ce.valuenum IS NOT NULL
          AND ce.charttime BETWEEN ie2.intime
                               AND DATETIME_ADD(ie2.intime, INTERVAL 24 HOUR)
        GROUP BY ce.stay_id, ce.charttime
        ORDER BY ce.stay_id, ce.charttime
        """
        chunk_df = run_bq(query, verbose=False)
        results.append(chunk_df)

    df_vitals = pd.concat(results, ignore_index=True)
    return df_vitals


TEMPORAL_FEATURE_COLS = [
    'age', 'gender_num', 'bmi',
    'heart_rate', 'resp_rate', 'spo2', 'mbp', 'temperature', 'fio2', 'peep',
    'pao2', 'paco2', 'ph', 'pf_ratio',
    'lactate', 'creatinine', 'bilirubin', 'platelets', 'wbc',
    'sofa_score', 'gcs_total',
    'hours_since_admit', 'vasopressor_flag',
]
N_TEMPORAL_COVARIATES = len(TEMPORAL_FEATURE_COLS)


def build_temporal_sequences(df_vitals, df_cohort, seq_len=48):
    """Build padded time series tensors at 0.5h resolution for Transformer."""
    def _pad_sequences(seqs, maxlen, dtype='float32', padding='pre', value=0.0):
        n_feat = seqs[0].shape[1] if len(seqs) > 0 else 0
        out = np.full((len(seqs), maxlen, n_feat), value, dtype=dtype)
        for i, s in enumerate(seqs):
            trunc = s[-maxlen:] if len(s) > maxlen else s
            if padding == 'pre':
                out[i, maxlen - len(trunc):, :] = trunc
            else:
                out[i, :len(trunc), :] = trunc
        return out

    df_merged = df_vitals.merge(
        df_cohort[['stay_id', 'age', 'gender', 'icu_intime',
                    'Treatment_W', 'vfd28', 'delta']].drop_duplicates('stay_id'),
        on='stay_id', how='inner'
    )

    if 'sofa_X' in df_cohort.columns:
        sofa_map = df_cohort.set_index('stay_id')['sofa_X'].to_dict()
        df_merged['sofa_score'] = df_merged['stay_id'].map(sofa_map)
    else:
        df_merged['sofa_score'] = np.nan

    df_merged['gender_num'] = (df_merged['gender'] == 'M').astype(float)
    df_merged['hours_since_admit'] = (
        pd.to_datetime(df_merged['charttime']) -
        pd.to_datetime(df_merged['icu_intime'])
    ).dt.total_seconds() / 3600

    if 'bmi_X' in df_cohort.columns:
        bmi_map = df_cohort.drop_duplicates('stay_id').set_index('stay_id')['bmi_X'].to_dict()
        df_merged['bmi'] = df_merged['stay_id'].map(bmi_map)
    else:
        df_merged['bmi'] = np.nan
    if 'gcs_X' in df_cohort.columns:
        gcs_map = df_cohort.drop_duplicates('stay_id').set_index('stay_id')['gcs_X'].to_dict()
        df_merged['gcs_total'] = df_merged['stay_id'].map(gcs_map)
        _miss_gcs = df_merged['gcs_total'].isna()
        if _miss_gcs.any():
            df_merged.loc[_miss_gcs, 'gcs_total'] = (
                df_merged.loc[_miss_gcs, 'gcs_component'] * 3.0)
    else:
        df_merged['gcs_total'] = df_merged['gcs_component'] * 3.0
    df_merged['vasopressor_flag'] = 0.0

    df_merged['pf_ratio'] = np.nan

    sequences, treatments, vfd_list, delta_list, valid_ids = [], [], [], [], []

    _iPF   = TEMPORAL_FEATURE_COLS.index('pf_ratio')
    _iPaO2 = TEMPORAL_FEATURE_COLS.index('pao2')
    _iFiO2 = TEMPORAL_FEATURE_COLS.index('fio2')
    _iSpO2 = TEMPORAL_FEATURE_COLS.index('spo2')

    for stay_id, group in df_merged.groupby('stay_id'):
        group = group.sort_values('charttime')
        group = group.set_index('charttime')
        group_resampled = group[TEMPORAL_FEATURE_COLS].resample('30min').mean()
        group_resampled = group_resampled.ffill().bfill()
        vals = np.asarray(group_resampled.values, dtype=np.float64)

        _fio2_frac = np.clip(vals[:, _iFiO2] / 100.0, 0.21, 1.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            _pf = vals[:, _iPaO2] / _fio2_frac
            _sf = vals[:, _iSpO2] / _fio2_frac
            _pf_from_sf = (_sf - 64.0) / 0.84
        _use_sf = (~np.isfinite(_pf)) & np.isfinite(_pf_from_sf) & (vals[:, _iSpO2] <= 97)
        _pf = np.where(_use_sf, _pf_from_sf, _pf)
        _pf = np.where(np.isfinite(_pf) & (_pf > 0), _pf, np.nan)
        vals[:, _iPF] = _pf
        if len(vals) < 2:
            continue
        sequences.append(vals)
        treatments.append(group['Treatment_W'].iloc[0])
        vfd_list.append(group['vfd28'].iloc[0])
        delta_list.append(group['delta'].iloc[0])
        valid_ids.append(stay_id)

    X = _pad_sequences(sequences, maxlen=seq_len, dtype='float32',
                       padding='pre', value=np.nan)
    W = np.array(treatments, dtype=np.float32)
    VFD = np.array(vfd_list, dtype=np.float32)
    D = np.array(delta_list, dtype=np.float32)

    return X, W, VFD, D, valid_ids


def propensity_score_match(X_all, W_all, caliper_scale=0.1, random_state=42):
    """PSM with logit-caliper on baseline covariates."""
    X_baseline = X_all[:, -1, :]
    X_baseline_clean = np.nan_to_num(X_baseline, nan=0.0)

    scaler = StandardScaler()
    X_baseline_scaled = scaler.fit_transform(X_baseline_clean)

    ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
    ps_model.fit(X_baseline_scaled, W_all)

    ps = ps_model.predict_proba(X_baseline_scaled)[:, 1]
    logit_ps = np.log(ps / (1 - ps + 1e-8))
    caliper = caliper_scale * logit_ps.std()

    nirs_idx = np.where(W_all == 1)[0]
    imv_idx = np.where(W_all == 0)[0]

    rng_m = np.random.RandomState(random_state)
    order = rng_m.permutation(len(nirs_idx))
    ctrl_logit = logit_ps[imv_idx]
    available = np.ones(len(imv_idx), dtype=bool)
    mn, mi = [], []
    for k in order:
        ti = nirs_idx[k]
        d = np.abs(ctrl_logit - logit_ps[ti])
        d[~available] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper:
            mn.append(ti); mi.append(imv_idx[j]); available[j] = False
    matched_nirs = np.array(mn, dtype=int)
    matched_imv = np.array(mi, dtype=int)

    psm_idx = np.concatenate([matched_nirs, matched_imv])
    np.random.RandomState(random_state).shuffle(psm_idx)

    assert len(np.unique(psm_idx)) == len(psm_idx), "duplicate rows after matching"
    return psm_idx, ps_model, ps


PHYSIOLOGIC_RANGES = {
    'heart_rate': (20, 220), 'resp_rate': (4, 60), 'spo2': (50, 100),
    'mbp': (30, 150), 'temperature': (32, 42), 'fio2': (21, 100),
    'peep': (0, 25), 'pao2': (30, 600), 'paco2': (15, 120),
    'ph': (6.8, 7.8), 'pf_ratio': (30, 600), 'lactate': (0.2, 20),
    'creatinine': (0.1, 15), 'bilirubin': (0.1, 40), 'platelets': (5, 1000),
    'wbc': (0.1, 100), 'sofa_score': (0, 24), 'gcs_total': (3, 15),
    'age': (18, 100), 'bmi': (10, 70),
}


def clean_and_impute_temporal(X, feature_names=None, add_missing_indicators=False,
                              verbose=True, reference_medians=None):
    """Clip implausible values, then impute without fabricating zeros."""
    if feature_names is None:
        feature_names = TEMPORAL_FEATURE_COLS
    X = np.asarray(X, dtype=np.float64).copy()
    N, T, D = X.shape
    nan_before = np.isnan(X).mean()

    n_clipped = 0
    for j, nm in enumerate(feature_names[:D]):
        if nm in PHYSIOLOGIC_RANGES:
            lo, hi = PHYSIOLOGIC_RANGES[nm]
            col = X[:, :, j]
            bad = (~np.isnan(col)) & ((col < lo) | (col > hi))
            n_clipped += int(bad.sum())
            col[bad] = np.nan
            X[:, :, j] = col

    for j in range(D):
        col = X[:, :, j]
        idx = np.where(~np.isnan(col), np.arange(T)[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        col_f = col[np.arange(N)[:, None], idx]
        rev = col_f[:, ::-1]
        idxb = np.where(~np.isnan(rev), np.arange(T)[None, :], 0)
        np.maximum.accumulate(idxb, axis=1, out=idxb)
        col_b = rev[np.arange(N)[:, None], idxb][:, ::-1]
        X[:, :, j] = col_b

    med_used = {}
    for j, nm in enumerate(feature_names[:D]):
        col = X[:, :, j]
        if np.isnan(col).any():
            if reference_medians is not None and nm in reference_medians:
                med = float(reference_medians[nm])
            else:
                vals = col[~np.isnan(col)]
                med = float(np.median(vals)) if vals.size else 0.0
            med_used[nm] = med
            col[np.isnan(col)] = med
            X[:, :, j] = col

    all_med = {}
    for j, nm in enumerate(feature_names[:D]):
        v = X[:, :, j]
        v = v[np.isfinite(v)]
        all_med[nm] = float(np.median(v)) if v.size else 0.0
    info = {'nan_before': nan_before, 'nan_after': float(np.isnan(X).mean()),
            'n_range_clipped': n_clipped, 'median_filled': med_used,
            'medians': all_med,
            'medians_source': 'reference' if reference_medians is not None else 'self'}

    if add_missing_indicators:
        raise ValueError(
            "add_missing_indicators is handled by normalize_and_mask(), which has "
            "the pre-imputation tensor. Call normalize_and_mask(..., "
            "add_missing_indicators=True) instead.")
    return X.astype(np.float32), info


def normalize_and_mask(X_psm, n_covariates, add_missing_indicators=False,
                       robust=True, feature_names=None, reference_scaler=None):
    """Clean -> impute -> robustly scale, and build true pad masks."""
    if feature_names is None:
        feature_names = TEMPORAL_FEATURE_COLS
    X_raw = np.asarray(X_psm, dtype=np.float64)
    N, T, D = X_raw.shape

    miss = np.isnan(X_raw).copy()
    for j, nm in enumerate(feature_names[:D]):
        if nm in PHYSIOLOGIC_RANGES:
            lo, hi = PHYSIOLOGIC_RANGES[nm]
            col = X_raw[:, :, j]
            miss[:, :, j] |= (~np.isnan(col)) & ((col < lo) | (col > hi))

    _ref_med = (reference_scaler.get('medians')
                if reference_scaler is not None else None)
    X_imp, info = clean_and_impute_temporal(
        X_raw, feature_names=feature_names, add_missing_indicators=False,
        reference_medians=_ref_med)

    Xf = X_imp.reshape(-1, D).astype(np.float64)
    if reference_scaler is not None:
        centre = np.asarray(reference_scaler['centre'], dtype=np.float64)
        scale = np.asarray(reference_scaler['scale'], dtype=np.float64).copy()
        scale[scale < 1e-8] = 1.0
        if centre.shape[0] != D:
            raise ValueError(f"reference_scaler has {centre.shape[0]} covariates "
                             f"but this tensor has {D}")
    elif robust:
        centre = np.median(Xf, axis=0)
        q75, q25 = np.percentile(Xf, [75, 25], axis=0)
        scale = (q75 - q25)
        scale[scale < 1e-8] = 1.0
    else:
        centre = Xf.mean(axis=0)
        scale = Xf.std(axis=0); scale[scale < 1e-8] = 1.0
    X_scaled = ((Xf - centre) / scale).reshape(N, T, D)
    X_scaled = np.clip(X_scaled, -10, 10)

    pad_masks = miss.all(axis=-1)

    if add_missing_indicators:
        X_scaled = np.concatenate([X_scaled, miss.astype(np.float64)], axis=-1)

    ts_scaler = {'centre': centre, 'scale': scale, 'robust': robust,
                 'feature_names': list(feature_names[:D]),
                 'add_missing_indicators': add_missing_indicators,
                 'impute_info': info,
                 'medians': (reference_scaler['medians']
                             if reference_scaler is not None
                             and 'medians' in reference_scaler
                             else info.get('medians'))}
    return X_scaled.astype(np.float32), pad_masks, ts_scaler


def extract_eicu_cohort():

    sql = f"""
    SELECT
        p.patientunitstayid   AS stay_id,
        p.uniquepid            AS subject_id,
        p.patienthealthsystemstayid AS hadm_id,
        p.hospitaladmitoffset,
        p.hospitaldischargeoffset,
        p.unitdischargeoffset,
        p.age,
        p.gender,
        p.unitdischargestatus,
        p.hospitaldischargestatus,
        p.unittype,
        p.unitdischargeoffset / (60.0 * 24.0) AS los_icu_days,
        0 AS icu_intime_offset
    FROM `{EICU['main']}.patient` p
    WHERE p.age != ''
      AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
      AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
      AND p.unitvisitNumber = 1
    """
    df = run_bq(sql)
    df["age_clean"] = df["age"].str.replace("> ", "", regex=False)
    df["age_clean"] = pd.to_numeric(df["age_clean"], errors="coerce")
    df = df[df["age_clean"] >= 18].copy()
    return df


def assign_eicu_treatment(df_eicu):
    """eICU treatment assignment using ONLY ventilation_events with start/end pairing."""
    _elig = f"""
        SELECT p.patientunitstayid
        FROM `{EICU['main']}.patient` p
        WHERE p.age != ''
          AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
          AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
          AND p.unitvisitNumber = 1
    """

    sql_vent_all = f"""
    WITH eligible AS ({_elig})
    SELECT
        ve.patientunitstayid AS stay_id,
        ve.event,
        ve.hrs
    FROM `{EICU['derived']}.ventilation_events` ve
    JOIN eligible e ON ve.patientunitstayid = e.patientunitstayid
    WHERE ve.event IN (
        'mechvent start', 'Trach', 'niv start',
        'mechvent end', 'niv end'
    )
    """

    df_vent = run_bq(sql_vent_all)

    start_map = {
        'mechvent start': 'Invasive',
        'Trach': 'Invasive',
        'niv start': 'NIV',
    }
    end_map = {
        'mechvent end': 'Invasive',
        'niv end': 'NIV',
    }

    df_s = df_vent[df_vent['event'].isin(start_map.keys())].copy()
    df_s['vent_type'] = df_s['event'].map(start_map)
    df_s['action'] = 'start'

    df_e = df_vent[df_vent['event'].isin(end_map.keys())].copy()
    df_e['vent_type'] = df_e['event'].map(end_map)
    df_e['action'] = 'end'

    df_all_events = pd.concat([df_s, df_e]).sort_values(
        ['stay_id', 'vent_type', 'hrs'])

    def _get_valid_start(group):
        """Return earliest valid start time for episodes extending into ICU (hrs>0)."""
        valid_starts = []
        current_start = None
        for row in group.itertuples():
            if row.action == 'start':
                if current_start is None:
                    current_start = row.hrs
            elif row.action == 'end':
                if current_start is not None:
                    if row.hrs > 0:
                        valid_starts.append(current_start)
                    current_start = None
        if current_start is not None:
            valid_starts.append(current_start)
        return min(valid_starts) if valid_starts else np.nan

    df_filtered = (df_all_events
                   .groupby(['stay_id', 'vent_type'])
                   .apply(_get_valid_start)
                   .rename('first_hrs'))
    df_first_events = df_filtered.unstack(level='vent_type')

    def _categorize(row):
        first_inv = row.get('Invasive', np.nan)
        first_niv = row.get('NIV', np.nan)
        inv_exists = pd.notna(first_inv)
        niv_exists = pd.notna(first_niv)

        if inv_exists and not niv_exists:
            return 'IMV_only', 0.0
        if not inv_exists and niv_exists:
            return 'NIRS_only', 1.0
        if not inv_exists and not niv_exists:
            return 'exclude', np.nan
        if first_inv < first_niv:
            return 'IMV_then_NIRS', 0.0
        if first_niv < first_inv:
            return 'NIRS_then_IMV', 1.0
        return 'ambiguous', np.nan

    results = df_first_events.apply(_categorize, axis=1)
    df_tx = pd.DataFrame({
        'stay_id': df_first_events.index,
        'category': [r[0] for r in results],
        'Treatment_W': [r[1] for r in results],
    })

    _first_h = df_first_events.min(axis=1)
    df_tx['hrs_to_support'] = df_tx['stay_id'].map(_first_h)
    _drop = ~(df_tx['hrs_to_support'] <= SUPPORT_WINDOW_HOURS)
    df_tx = df_tx[~_drop].copy()

    df_tx = df_tx[df_tx['Treatment_W'].notna()].copy()
    df_tx['Treatment_W'] = df_tx['Treatment_W'].astype(int)

    df_treat = df_tx[['stay_id', 'Treatment_W', 'category']]
    df_out = df_eicu.merge(df_treat, on='stay_id', how='inner')

    return df_out


EICU_ARF_ICD9 = ['518.81', '518.82', '518.84', '518.5', '799.1', '518.4']
EICU_ARF_STRINGS = [
    'respiratory failure', 'acute respiratory distress', 'ards',
    'pulmonary edema', 'acute lung injury', 'hypoxemia', 'hypercapnia',
]


def identify_eicu_arf(df_eicu):
    """Identify acute respiratory failure in eICU, mirroring MIMIC's"""
    stay_ids = df_eicu['stay_id'].astype(int).tolist()
    id_list = ','.join(map(str, stay_ids))

    like_clause = " OR ".join(
        [f"LOWER(d.diagnosisstring) LIKE '%{t}%'" for t in EICU_ARF_STRINGS])
    icd_clause = " OR ".join(
        [f"d.icd9code LIKE '%{c}%'" for c in EICU_ARF_ICD9])

    sql_dx = f"""
    SELECT DISTINCT d.patientunitstayid AS stay_id
    FROM `{EICU['main']}.diagnosis` d
    WHERE d.patientunitstayid IN ({id_list})
      AND ( {icd_clause} OR {like_clause} )
    """
    try:
        df_dx = run_bq(sql_dx)
    except Exception as e:
        sql_dx_str = f"""
        SELECT DISTINCT d.patientunitstayid AS stay_id
        FROM `{EICU['main']}.diagnosis` d
        WHERE d.patientunitstayid IN ({id_list})
          AND ( {like_clause} )
        """
        df_dx = run_bq(sql_dx_str)
    dx_ids = set(df_dx['stay_id'].astype(int))

    sql_phys = f"""
    WITH v AS (
      SELECT patientunitstayid AS stay_id, MIN(sao2) AS spo2_min
      FROM `{EICU['main']}.vitalperiodic`
      WHERE patientunitstayid IN ({id_list})
        AND observationoffset BETWEEN 0 AND 1440
        AND sao2 BETWEEN 50 AND 100
      GROUP BY 1
    ),
    vn AS (
      SELECT patientunitstayid AS stay_id,
             MIN(SAFE_CAST(nursingchartvalue AS FLOAT64)) AS spo2_min_nc
      FROM `{EICU['main']}.nursecharting`
      WHERE patientunitstayid IN ({id_list})
        AND nursingchartoffset BETWEEN 0 AND 1440
        AND nursingchartcelltypevalname = 'O2 Saturation'
        AND SAFE_CAST(nursingchartvalue AS FLOAT64) BETWEEN 50 AND 100
      GROUP BY 1
    ),
    g AS (
      SELECT patientunitstayid AS stay_id,
             MIN(CASE WHEN LOWER(labname) = 'pao2'  THEN labresult END) AS pao2_min,
             MAX(CASE WHEN LOWER(labname) = 'paco2' THEN labresult END) AS paco2_max,
             MIN(CASE WHEN LOWER(labname) = 'ph'    THEN labresult END) AS ph_min
      FROM `{EICU['main']}.lab`
      WHERE patientunitstayid IN ({id_list})
        AND labresultoffset BETWEEN 0 AND 1440
      GROUP BY 1
    )
    SELECT COALESCE(v.stay_id, vn.stay_id, g.stay_id) AS stay_id,
           LEAST(COALESCE(v.spo2_min, 999), COALESCE(vn.spo2_min_nc, 999)) AS spo2_min,
           g.pao2_min, g.paco2_max, g.ph_min
    FROM v
    FULL OUTER JOIN vn ON v.stay_id = vn.stay_id
    FULL OUTER JOIN g  ON COALESCE(v.stay_id, vn.stay_id) = g.stay_id
    """
    try:
        df_ph = run_bq(sql_phys)
    except Exception as e:
        df_out = df_eicu[df_eicu['stay_id'].astype(int).isin(dx_ids)].copy()
        return df_out
    df_ph['spo2_min'] = df_ph['spo2_min'].replace(999, np.nan)
    ok = (
        (df_ph['spo2_min'] < 94)
        | (df_ph['pao2_min'] < 60)
        | ((df_ph['paco2_max'] > 50) & (df_ph['ph_min'] < 7.35))
    )
    phys_ids = set(df_ph.loc[ok.fillna(False), 'stay_id'].astype(int))

    keep = dx_ids & phys_ids
    df_out = df_eicu[df_eicu['stay_id'].astype(int).isin(keep)].copy()
    return df_out


def apply_eicu_exclusions(df_eicu, exclude_crash=True):
    """Apply MIMIC's three exclusions to eICU so the cohorts are comparable:"""
    stay_ids = df_eicu['stay_id'].astype(int).tolist()
    id_list = ','.join(map(str, stay_ids))

    sql_crash = f"""
    SELECT patientunitstayid AS stay_id, MIN(hrs) AS first_inv_hrs
    FROM `{EICU['derived']}.ventilation_events`
    WHERE patientunitstayid IN ({id_list})
      AND event IN ('mechvent start', 'Trach')
    GROUP BY 1
    HAVING MIN(hrs) < 1.0
    """
    if not exclude_crash:
        excl_crash = set()
    else:
      try:
        excl_crash = set(run_bq(sql_crash)['stay_id'].astype(int))
      except Exception as e:
        excl_crash = set()

    sql_trach = f"""
    SELECT DISTINCT patientunitstayid AS stay_id
    FROM `{EICU['derived']}.ventilation_events`
    WHERE patientunitstayid IN ({id_list})
      AND event = 'Trach' AND hrs <= 0
    """
    try:
        excl_trach = set(run_bq(sql_trach)['stay_id'].astype(int))
    except Exception as e:
        excl_trach = set()

    sql_chronic = f"""
    SELECT DISTINCT patientunitstayid AS stay_id
    FROM `{EICU['main']}.pasthistory`
    WHERE patientunitstayid IN ({id_list})
      AND ( LOWER(pasthistorypath)  LIKE '%home ventilator%'
         OR LOWER(pasthistorypath)  LIKE '%tracheostomy%'
         OR LOWER(pasthistoryvalue) LIKE '%home ventilator%'
         OR LOWER(pasthistoryvalue) LIKE '%tracheostomy%' )
    """
    try:
        excl_chronic = set(run_bq(sql_chronic)['stay_id'].astype(int))
    except Exception as e:
        excl_chronic = set()

    sql_dnr = f"""
    SELECT DISTINCT patientunitstayid AS stay_id
    FROM `{EICU['main']}.careplangeneral`
    WHERE patientunitstayid IN ({id_list})
      AND ( LOWER(cplitemvalue) LIKE '%do not resuscitate%'
         OR LOWER(cplitemvalue) LIKE '%no cpr%'
         OR LOWER(cplitemvalue) LIKE '%do not intubate%'
         OR LOWER(cplitemvalue) LIKE '%comfort measures%'
         OR LOWER(cplitemvalue) LIKE '%end of life%' )
    """
    try:
        excl_dnr = set(run_bq(sql_dnr)['stay_id'].astype(int))
    except Exception as e:
        excl_dnr = set()

    sql_died24 = f"""
    SELECT DISTINCT patientunitstayid AS stay_id
    FROM `{EICU['main']}.patient`
    WHERE patientunitstayid IN ({id_list})
      AND ( (LOWER(unitdischargestatus)     = 'expired'
             AND unitdischargeoffset     <= 1440)
         OR (LOWER(hospitaldischargestatus) = 'expired'
             AND hospitaldischargeoffset <= 1440) )
    """
    if not EXCLUDE_DIED_24H:
        excl_died24 = set()
    else:
        try:
            excl_died24 = set(run_bq(sql_died24)['stay_id'].astype(int))
        except Exception as e:
            excl_died24 = set()

    all_excl = excl_crash | excl_trach | excl_chronic | excl_dnr | excl_died24
    df_out = df_eicu[~df_eicu['stay_id'].astype(int).isin(all_excl)].copy()
    return df_out

def extract_eicu_covariates(df_eicu):
    """23 covariates from eICU, harmonized to MIMIC naming."""
    _elig = f"""
        SELECT p.patientunitstayid
        FROM `{EICU['main']}.patient` p
        WHERE p.age != ''
          AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
          AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
          AND p.unitvisitNumber = 1
    """

    sql_demo = f"""
    WITH eligible AS ({_elig})
    SELECT
        p.patientunitstayid AS stay_id,
        SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) AS age_X,
        CASE WHEN p.gender = 'Male' THEN 1 ELSE 0 END AS gender_X,
        CASE WHEN p.admissionheight > 0 AND p.admissionweight > 0
             THEN p.admissionweight / POW(p.admissionheight / 100.0, 2)
             ELSE NULL END AS bmi_X,
        NULLIF(apr.acutephysiologyscore, -1) AS apache_aps,
        CASE WHEN apv.eyes   >= 1 AND apv.motor  >= 1 AND apv.verbal >= 1
             THEN apv.eyes + apv.motor + apv.verbal
             ELSE NULL END AS gcs_X
    FROM `{EICU['main']}.patient` p
    JOIN eligible e ON p.patientunitstayid = e.patientunitstayid
    LEFT JOIN (
        SELECT patientunitstayid, MAX(acutephysiologyscore) AS acutephysiologyscore
        FROM `{EICU['main']}.apachepatientresult`
        GROUP BY patientunitstayid
    ) apr ON p.patientunitstayid = apr.patientunitstayid
    LEFT JOIN `{EICU['main']}.apacheapsvar` apv
        ON p.patientunitstayid = apv.patientunitstayid
    """

    sql_vitals = f"""
    WITH eligible AS ({_elig})
    SELECT
        nc.patientunitstayid AS stay_id,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Heart Rate'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS hr_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Respiratory Rate'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS rr_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'O2 Saturation'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS spo2_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Non-Invasive BP Mean'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS mbp_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Temperature (C)'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS tempc_mean_X
    FROM `{EICU['main']}.nursecharting` nc
    JOIN eligible e ON nc.patientunitstayid = e.patientunitstayid
    WHERE nc.nursingchartoffset >= 0
      AND nc.nursingchartoffset <= {T0_WINDOW_H * 60}
    GROUP BY nc.patientunitstayid
    """

    sql_labs = f"""
    WITH eligible AS ({_elig})
    SELECT
        l.patientunitstayid AS stay_id,
        AVG(CASE WHEN LOWER(l.labname) = 'pao2' THEN l.labresult END) AS pao2_X,
        AVG(CASE WHEN LOWER(l.labname) = 'paco2' THEN l.labresult END) AS paco2_X,
        AVG(CASE WHEN LOWER(l.labname) = 'ph' THEN l.labresult END) AS ph_X,
        AVG(CASE WHEN LOWER(l.labname) IN ('fio2', 'fio2 (%)')
                 THEN l.labresult END) AS fio2_X,
        AVG(CASE WHEN LOWER(l.labname) = 'lactate' THEN l.labresult END) AS lactate_X,
        AVG(CASE WHEN LOWER(l.labname) IN ('bicarbonate', 'hco3')
                 THEN l.labresult END) AS bicarbonate_X
    FROM `{EICU['main']}.lab` l
    JOIN eligible e ON l.patientunitstayid = e.patientunitstayid
    WHERE l.labresultoffset >= 0
      AND l.labresultoffset <= {T0_WINDOW_H * 60}
    GROUP BY l.patientunitstayid
    """

    sql_comor = f"""
    WITH eligible AS ({_elig})
    SELECT
        p.patientunitstayid AS stay_id,
        MAX(CASE WHEN LOWER(ph.pasthistorypath) LIKE '%copd%'
                   OR LOWER(ph.pasthistorypath) LIKE '%chronic obstructive%'
            THEN 1 ELSE 0 END) AS copd_X,
        MAX(CASE WHEN LOWER(ph.pasthistorypath) LIKE '%chf%'
                   OR LOWER(ph.pasthistorypath) LIKE '%heart failure%'
                   OR LOWER(ph.pasthistorypath) LIKE '%congestive%'
            THEN 1 ELSE 0 END) AS chf_X,
        MAX(CASE WHEN LOWER(ph.pasthistorypath) LIKE '%immunosuppre%'
                   OR LOWER(ph.pasthistorypath) LIKE '%aids%'
                   OR LOWER(ph.pasthistorypath) LIKE '%transplant%'
                   OR LOWER(ph.pasthistorypath) LIKE '%chemotherapy%'
            THEN 1 ELSE 0 END) AS immunosuppressed_X
    FROM `{EICU['main']}.patient` p
    JOIN eligible e ON p.patientunitstayid = e.patientunitstayid
    LEFT JOIN `{EICU['main']}.pasthistory` ph
        ON p.patientunitstayid = ph.patientunitstayid
    GROUP BY p.patientunitstayid
    """

    sql_sepsis = f"""
    WITH eligible AS ({_elig})
    SELECT DISTINCT d.patientunitstayid AS stay_id, 1 AS sepsis_X
    FROM `{EICU['main']}.diagnosis` d
    JOIN eligible e ON d.patientunitstayid = e.patientunitstayid
    WHERE LOWER(d.diagnosisstring) LIKE '%sepsis%'
       OR LOWER(d.diagnosisstring) LIKE '%septic%'
    """

    df_demo   = run_bq(sql_demo)
    df_vitals = run_bq(sql_vitals)
    df_labs   = run_bq(sql_labs)
    df_comor  = run_bq(sql_comor)
    df_sepsis = run_bq(sql_sepsis)

    df = df_demo
    for dfi in [df_vitals, df_labs, df_comor, df_sepsis]:
        df = df.merge(dfi, on="stay_id", how="left")

    if "apache_aps" in df.columns:
        df["sofa_X"] = np.clip(df["apache_aps"] / 10.0, 0, 24)
    else:
        df["sofa_X"] = np.nan
    df["sapsii_X"] = df.get("apache_aps", np.nan)

    if "fio2_X" in df.columns:
        mask_frac = df["fio2_X"] <= 1.0
        df.loc[mask_frac, "fio2_X"] = df.loc[mask_frac, "fio2_X"] * 100.0

    for col in ["pao2_X", "fio2_X", "spo2_mean_X", "rr_mean_X"]:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                med = df[col].median()
                df[col] = df[col].fillna(med)

    df["pf_ratio_X"] = safe_divide(
        df["pao2_X"].values, df["fio2_X"].values / 100.0)
    df["pf_ratio_X"] = np.clip(df["pf_ratio_X"], 0, 700)

    df["rox_index_X"] = safe_divide(
        safe_divide(df["spo2_mean_X"].values, df["fio2_X"].values / 100.0),
        df["rr_mean_X"].values)
    df["rox_index_X"] = np.clip(df["rox_index_X"], 0, 30)

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df.drop(columns=["apache_aps"], inplace=True, errors="ignore")

    return df


def compute_eicu_vfd28(df_eicu):
    """VFD-28 for eICU using ventilation_events state machine."""
    _elig = f"""
        SELECT p.patientunitstayid
        FROM `{EICU['main']}.patient` p
        WHERE p.age != ''
          AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
          AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
          AND p.unitvisitNumber = 1
    """

    sql_mechvent = f"""
    WITH eligible AS ({_elig})
    SELECT ve.patientunitstayid AS stay_id, ve.event, ve.hrs
    FROM `{EICU['derived']}.ventilation_events` ve
    JOIN eligible e ON ve.patientunitstayid = e.patientunitstayid
    WHERE ve.event IN ('mechvent start', 'mechvent end', 'Trach')
      AND ve.hrs <= {VFD_HORIZON_DAYS * 24}
    ORDER BY ve.patientunitstayid, ve.hrs
    """

    _horizon_min = VFD_HORIZON_DAYS * 24 * 60
    sql_death = f"""
    SELECT
        p.patientunitstayid AS stay_id,
        CASE
            WHEN p.hospitaldischargestatus = 'Expired'
                 AND p.hospitaldischargeoffset <= {_horizon_min} THEN 1
            WHEN p.unitdischargestatus = 'Expired'
                 AND p.unitdischargeoffset <= {_horizon_min} THEN 1
            ELSE 0
        END AS died_28d,
        CASE
            WHEN p.hospitaldischargestatus = 'Expired' THEN 1
            WHEN p.unitdischargestatus = 'Expired' THEN 1
            ELSE 0
        END AS died_any_time,
        p.hospitaldischargeoffset / (60.0 * 24.0) AS days_to_discharge
    FROM `{EICU['main']}.patient` p
    WHERE p.age != ''
      AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
      AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
      AND p.unitvisitNumber = 1
    """

    df_mechvent = run_bq(sql_mechvent)
    df_death    = run_bq(sql_death)

    horizon_hrs = VFD_HORIZON_DAYS * 24

    def _imv_hours(grp):
        grp = grp.sort_values("hrs")
        total, on, t0 = 0.0, False, 0.0
        for _, row in grp.iterrows():
            if row["event"] in ("mechvent start", "Trach") and not on:
                on, t0 = True, max(row["hrs"], 0)
            elif row["event"] == "mechvent end" and on:
                total += row["hrs"] - t0
                on = False
        if on:
            total += horizon_hrs - t0
        return total

    if len(df_mechvent) > 0:
        imv_series = df_mechvent.groupby("stay_id").apply(_imv_hours)
        df_imv_hrs = imv_series.reset_index()
        df_imv_hrs.columns = ["stay_id", "total_imv_hours"]
    else:
        df_imv_hrs = pd.DataFrame(columns=["stay_id", "total_imv_hours"])

    df = df_eicu[["stay_id", "Treatment_W"]].copy()
    df = df.merge(df_imv_hrs, on="stay_id", how="left")
    df = df.merge(df_death, on="stay_id", how="left")

    df["total_imv_hours"] = df["total_imv_hours"].fillna(0)
    df["died_28d"] = df["died_28d"].fillna(0).astype(int)
    df["total_imv_days"] = df["total_imv_hours"] / 24.0
    df["vfd28"] = np.where(
        df["died_28d"] == 1, 0.0,
        np.clip(28.0 - df["total_imv_days"], 0, 28))
    df["delta"] = 1 - df["died_28d"]

    return df


def propensity_score_match_baseline(X_baseline, W, feature_cols,
                                     caliper_scale=0.1, random_state=42):
    """PSM on baseline (non-temporal) covariates for eICU external validation."""
    X_clean = np.nan_to_num(X_baseline, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
    ps_model.fit(X_scaled, W)

    ps = ps_model.predict_proba(X_scaled)[:, 1]
    logit_ps = np.log(ps / (1 - ps + 1e-8))
    caliper = caliper_scale * logit_ps.std()

    nirs_idx = np.where(W == 1)[0]
    imv_idx = np.where(W == 0)[0]

    rng_m = np.random.RandomState(random_state)
    order = rng_m.permutation(len(nirs_idx))
    ctrl_logit = logit_ps[imv_idx]
    available = np.ones(len(imv_idx), dtype=bool)
    matched_nirs_l, matched_imv_l = [], []
    for k in order:
        ti = nirs_idx[k]
        d = np.abs(ctrl_logit - logit_ps[ti])
        d[~available] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper:
            matched_nirs_l.append(ti)
            matched_imv_l.append(imv_idx[j])
            available[j] = False
    matched_nirs = np.array(matched_nirs_l, dtype=int)
    matched_imv = np.array(matched_imv_l, dtype=int)

    psm_idx = np.concatenate([matched_nirs, matched_imv])
    np.random.RandomState(random_state).shuffle(psm_idx)

    assert len(np.unique(psm_idx)) == len(psm_idx), "matching produced duplicate rows"
    return psm_idx, ps_model, ps


def run_mimic_extraction(client=None, dataset_id=None, seq_len=48,
                         chunk_size=5000):
    """Full MIMIC-IV extraction pipeline — cohort, VFD-28, covariates,"""
    if client is not None:
        global _client
        _client = client

    print("Extracting MIMIC-IV cohort...")

    df_cohort = build_cohort()

    df_vfd = compute_vfd28(df_cohort)

    df_cohort = df_cohort.merge(
        df_vfd[["stay_id", "vfd28", "delta", "died_28d"]],
        on="stay_id", how="inner")

    df_cov = extract_baseline_covariates(df_cohort)

    _carry = [c for c in ['sofa_X', 'bmi_X', 'gcs_X'] if c in df_cov.columns]
    if _carry:
        df_cohort = df_cohort.merge(
            df_cov[['stay_id'] + _carry], on='stay_id', how='left')

    df_vitals = extract_temporal_chartevents(df_cohort, chunk_size)

    print("Building sequences...")
    X, W, VFD, D, valid_ids = build_temporal_sequences(
        df_vitals, df_cohort, seq_len=seq_len)

    return {
        'X': X, 'W': W, 'VFD': VFD, 'D': D,
        'valid_ids': valid_ids,
        'df_cohort': df_cohort,
        'df_vfd': df_vfd,
        'df_cov': df_cov,
        'df_cov_raw': LAST_BASELINE_RAW,
    }


def extract_eicu_temporal(df_eicu_tx, chunk_size=5000):
    """Extract time-series covariates from eICU for Transformer input."""
    stay_ids = df_eicu_tx['stay_id'].tolist()
    chunks = [stay_ids[i:i + chunk_size]
              for i in range(0, len(stay_ids), chunk_size)]

    print("Extracting eICU data...")

    results_nc = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_nc = f"""
        SELECT
            nc.patientunitstayid AS stay_id,
            CAST(FLOOR(nc.nursingchartoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Heart Rate'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS heart_rate,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Respiratory Rate'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS resp_rate,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'O2 Saturation'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS spo2,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Non-Invasive BP Mean'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS mbp,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Temperature (C)'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS temperature
        FROM `{EICU['main']}.nursecharting` nc
        WHERE nc.patientunitstayid IN ({chunk_str})
          AND nc.nursingchartoffset >= 0
          AND nc.nursingchartoffset <= 1440
        GROUP BY nc.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_nc, verbose=False)
        results_nc.append(chunk_df)

    df_nc = pd.concat(results_nc, ignore_index=True) if results_nc else pd.DataFrame()

    results_vp = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_vp = f"""
        SELECT
            vp.patientunitstayid AS stay_id,
            CAST(FLOOR(vp.observationoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(vp.heartrate)       AS heart_rate_vp,
            AVG(vp.sao2)            AS spo2_vp,
            AVG(vp.systemicmean)    AS mbp_vp,
            AVG(vp.temperature)     AS temperature_vp
        FROM `{EICU['main']}.vitalperiodic` vp
        WHERE vp.patientunitstayid IN ({chunk_str})
          AND vp.observationoffset >= 0
          AND vp.observationoffset <= 1440
        GROUP BY vp.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_vp, verbose=False)
        results_vp.append(chunk_df)

    df_vp = pd.concat(results_vp, ignore_index=True) if results_vp else pd.DataFrame()

    results_lab = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_lab = f"""
        SELECT
            l.patientunitstayid AS stay_id,
            CAST(FLOOR(l.labresultoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(CASE WHEN LOWER(l.labname) IN ('pao2', 'paO2')
                     THEN l.labresult END) AS pao2,
            AVG(CASE WHEN LOWER(l.labname) IN ('paco2', 'paCO2')
                     THEN l.labresult END) AS paco2,
            AVG(CASE WHEN LOWER(l.labname) IN ('ph', 'pH')
                     THEN l.labresult END) AS ph,
            AVG(CASE WHEN LOWER(l.labname) IN ('lactate', 'lactic acid')
                     THEN l.labresult END) AS lactate,
            AVG(CASE WHEN LOWER(l.labname) = 'creatinine'
                     THEN l.labresult END) AS creatinine,
            AVG(CASE WHEN LOWER(l.labname) IN ('total bilirubin', 'bilirubin')
                     THEN l.labresult END) AS bilirubin,
            AVG(CASE WHEN LOWER(l.labname) = 'platelets x 1000'
                     THEN l.labresult END) AS platelets,
            AVG(CASE WHEN LOWER(l.labname) IN ('wbc x 1000', 'wbc')
                     THEN l.labresult END) AS wbc
        FROM `{EICU['main']}.lab` l
        WHERE l.patientunitstayid IN ({chunk_str})
          AND l.labresultoffset >= 0
          AND l.labresultoffset <= 1440
        GROUP BY l.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_lab, verbose=False)
        results_lab.append(chunk_df)

    df_lab = pd.concat(results_lab, ignore_index=True) if results_lab else pd.DataFrame()

    results_resp = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_resp = f"""
        SELECT
            rc.patientunitstayid AS stay_id,
            CAST(FLOOR(rc.respchartoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(CASE WHEN LOWER(rc.respchartvaluelabel) LIKE '%fio2%'
                     THEN SAFE_CAST(rc.respchartvalue AS FLOAT64) END) AS fio2,
            AVG(CASE WHEN LOWER(rc.respchartvaluelabel) LIKE '%peep%'
                     THEN SAFE_CAST(rc.respchartvalue AS FLOAT64) END) AS peep
        FROM `{EICU['main']}.respiratorycharting` rc
        WHERE rc.patientunitstayid IN ({chunk_str})
          AND rc.respchartoffset >= 0
          AND rc.respchartoffset <= 1440
          AND SAFE_CAST(rc.respchartvalue AS FLOAT64) IS NOT NULL
        GROUP BY rc.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_resp, verbose=False)
        results_resp.append(chunk_df)

    df_resp = pd.concat(results_resp, ignore_index=True) if results_resp else pd.DataFrame()

    all_stay_ids = df_eicu_tx['stay_id'].unique()
    bins = np.arange(0, 1440, 30)
    idx = pd.MultiIndex.from_product([all_stay_ids, bins],
                                      names=['stay_id', 'offset_bin'])
    df_temporal = pd.DataFrame(index=idx).reset_index()

    if len(df_nc) > 0:
        df_temporal = df_temporal.merge(
            df_nc, on=['stay_id', 'offset_bin'], how='left')

    if len(df_vp) > 0:
        df_temporal = df_temporal.merge(
            df_vp, on=['stay_id', 'offset_bin'], how='left')
        for col, vp_col in [('heart_rate', 'heart_rate_vp'),
                             ('spo2', 'spo2_vp'),
                             ('mbp', 'mbp_vp'),
                             ('temperature', 'temperature_vp')]:
            if col in df_temporal.columns and vp_col in df_temporal.columns:
                df_temporal[col] = df_temporal[col].fillna(df_temporal[vp_col])
                df_temporal.drop(columns=[vp_col], inplace=True)

    if len(df_lab) > 0:
        df_temporal = df_temporal.merge(
            df_lab, on=['stay_id', 'offset_bin'], how='left')

    if len(df_resp) > 0:
        df_temporal = df_temporal.merge(
            df_resp, on=['stay_id', 'offset_bin'], how='left')

    for col in ['heart_rate', 'resp_rate', 'spo2', 'mbp', 'temperature',
                'fio2', 'peep', 'pao2', 'paco2', 'ph', 'lactate',
                'creatinine', 'bilirubin', 'platelets', 'wbc']:
        if col not in df_temporal.columns:
            df_temporal[col] = np.nan

    if 'fio2' in df_temporal.columns:
        mask_frac = df_temporal['fio2'] <= 1.0
        df_temporal.loc[mask_frac, 'fio2'] = df_temporal.loc[mask_frac, 'fio2'] * 100.0

    n_patients = df_temporal['stay_id'].nunique()
    n_rows = len(df_temporal)
    for col in ['heart_rate', 'resp_rate', 'spo2', 'fio2', 'pao2', 'lactate']:
        pct = df_temporal[col].notna().mean() * 100

    return df_temporal


def build_eicu_temporal_sequences(df_temporal, df_eicu_tx, df_eicu_vfd,
                                   df_eicu_cov, seq_len=48):
    """Build padded time series tensors from eICU temporal data."""
    def _pad_sequences(seqs, maxlen, dtype='float32', padding='pre', value=0.0):
        n_feat = seqs[0].shape[1] if len(seqs) > 0 else 0
        out = np.full((len(seqs), maxlen, n_feat), value, dtype=dtype)
        for i, s in enumerate(seqs):
            trunc = s[-maxlen:] if len(s) > maxlen else s
            if padding == 'pre':
                out[i, maxlen - len(trunc):, :] = trunc
            else:
                out[i, :len(trunc), :] = trunc
        return out

    static = df_eicu_tx[['stay_id']].drop_duplicates().copy()
    static = static.merge(
        df_eicu_tx[['stay_id', 'age_clean', 'gender']].drop_duplicates('stay_id'),
        on='stay_id', how='left')
    static = static.merge(
        df_eicu_cov[['stay_id', 'bmi_X', 'sofa_X', 'gcs_X']].drop_duplicates('stay_id'),
        on='stay_id', how='left')
    static = static.merge(
        df_eicu_vfd[['stay_id', 'Treatment_W', 'vfd28', 'died_28d']],
        on='stay_id', how='left')
    static['gender_num'] = (static['gender'] == 'Male').astype(float)
    static_dict = static.set_index('stay_id').to_dict('index')

    sequences, treatments, vfd_list, delta_list, valid_ids = [], [], [], [], []

    for stay_id, group in df_temporal.groupby('stay_id'):
        if stay_id not in static_dict:
            continue
        s = static_dict[stay_id]
        if pd.isna(s.get('Treatment_W')) or pd.isna(s.get('vfd28')):
            continue

        group = group.sort_values('offset_bin')

        vital_cols = ['heart_rate', 'resp_rate', 'spo2', 'mbp', 'temperature',
                      'fio2', 'peep', 'pao2', 'paco2', 'ph', 'lactate',
                      'creatinine', 'bilirubin', 'platelets', 'wbc']
        for col in vital_cols:
            if col in group.columns:
                group[col] = group[col].ffill().bfill()

        hours_since_admit = group['offset_bin'].values / 60.0
        fio2_vals = np.asarray(group['fio2'].values if 'fio2' in group.columns
                               else np.full(len(group), np.nan), dtype=np.float64)
        pao2_vals = np.asarray(group['pao2'].values if 'pao2' in group.columns
                               else np.full(len(group), np.nan), dtype=np.float64)

        fio2_frac = np.where(fio2_vals > 0, fio2_vals / 100.0, np.nan)
        fio2_frac = np.clip(fio2_frac, 0.21, 1.0)
        _spo2_vals = np.asarray(group['spo2'].values if 'spo2' in group.columns
                                else np.full(len(group), np.nan), dtype=np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            pf_ratio = np.where(np.isfinite(pao2_vals) & np.isfinite(fio2_frac),
                                pao2_vals / fio2_frac, np.nan)
            _sf = _spo2_vals / fio2_frac
            _pf_from_sf = (_sf - 64.0) / 0.84
        _use_sf = ((~np.isfinite(pf_ratio)) & np.isfinite(_pf_from_sf)
                   & (_spo2_vals <= 97))
        pf_ratio = np.where(_use_sf, _pf_from_sf, pf_ratio)
        pf_ratio = np.where(np.isfinite(pf_ratio) & (pf_ratio > 0), pf_ratio, np.nan)

        n_rows = len(group)
        feat_matrix = np.column_stack([
            np.full(n_rows, s.get('age_clean', np.nan)),
            np.full(n_rows, s.get('gender_num', 0)),
            np.full(n_rows, s.get('bmi_X', np.nan)),
            group['heart_rate'].values if 'heart_rate' in group.columns else np.full(n_rows, np.nan),
            group['resp_rate'].values if 'resp_rate' in group.columns else np.full(n_rows, np.nan),
            group['spo2'].values if 'spo2' in group.columns else np.full(n_rows, np.nan),
            group['mbp'].values if 'mbp' in group.columns else np.full(n_rows, np.nan),
            group['temperature'].values if 'temperature' in group.columns else np.full(n_rows, np.nan),
            fio2_vals,
            group['peep'].values if 'peep' in group.columns else np.full(n_rows, np.nan),
            pao2_vals,
            group['paco2'].values if 'paco2' in group.columns else np.full(n_rows, np.nan),
            group['ph'].values if 'ph' in group.columns else np.full(n_rows, np.nan),
            pf_ratio,
            group['lactate'].values if 'lactate' in group.columns else np.full(n_rows, np.nan),
            group['creatinine'].values if 'creatinine' in group.columns else np.full(n_rows, np.nan),
            group['bilirubin'].values if 'bilirubin' in group.columns else np.full(n_rows, np.nan),
            group['platelets'].values if 'platelets' in group.columns else np.full(n_rows, np.nan),
            group['wbc'].values if 'wbc' in group.columns else np.full(n_rows, np.nan),
            np.full(n_rows, s.get('sofa_X', np.nan)),
            np.full(n_rows, s.get('gcs_X', np.nan)),
            hours_since_admit,
            np.zeros(n_rows),
        ])

        feat_matrix = np.asarray(feat_matrix, dtype=np.float64)

        if feat_matrix.shape[0] < 2:
            continue

        sequences.append(feat_matrix)
        treatments.append(s['Treatment_W'])
        vfd_list.append(s['vfd28'])
        delta_list.append(1.0)
        valid_ids.append(stay_id)

    if len(sequences) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    X = _pad_sequences(sequences, maxlen=seq_len, dtype='float32',
                       padding='pre', value=np.nan)
    W = np.array(treatments, dtype=np.float32)
    VFD = np.array(vfd_list, dtype=np.float32)
    D = np.array(delta_list, dtype=np.float32)

    return X, W, VFD, D, valid_ids
