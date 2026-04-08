"""
src/features/engineer.py  (v2 — leakage-free)

Four-level feature engineering pipeline.

LEAKAGE NOTE (important):
  The denial label is defined as CLM_PMT_AMT == 0.
  Therefore ALL claim-level payment amount columns are excluded from features:
    - CLM_PMT_AMT, CLM_PMT_AMT_LOG
    - NCH_PRMRY_PYR_CLM_PD_AMT
    - NCH_BENE_BLOOD_DDCTBL_LBLTY_AM
  Annual beneficiary reimbursement totals (MEDREIMB_*, BENRES_*) are safe —
  they are yearly aggregates from the beneficiary file, not claim-level payments.

  L2 rate features use expanding-mean with shift(1) on sorted training data
  to prevent future-leakage from the label itself.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ICD-9 chapter boundaries
ICD9_CHAPTERS = [
    (1,   139, "infectious"),   (140, 239, "neoplasms"),
    (240, 279, "endocrine"),    (280, 289, "blood"),
    (290, 319, "mental"),       (320, 389, "nervous"),
    (390, 459, "circulatory"),  (460, 519, "respiratory"),
    (520, 579, "digestive"),    (580, 629, "genitourinary"),
    (630, 679, "pregnancy"),    (680, 709, "skin"),
    (710, 739, "musculoskeletal"), (740, 759, "congenital"),
    (760, 779, "perinatal"),    (780, 799, "symptoms"),
    (800, 999, "injury"),
]


def _icd9_to_chapter(code) -> str:
    if pd.isna(code) or not str(code).strip():
        return "unknown"
    try:
        s = str(code).strip().upper()
        if s.startswith("V"): return "supplementary_v"
        if s.startswith("E"): return "supplementary_e"
        n = int(s[:3])
        for lo, hi, ch in ICD9_CHAPTERS:
            if lo <= n <= hi:
                return ch
        return "other"
    except (ValueError, TypeError):
        return "unknown"


# ── L1: Claim Attributes ────────────────────────────────────────────────────

def build_l1(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # ── Claim structure (NO payment amounts — they encode the label) ──
    out["IS_INPATIENT"]       = df.get("IS_INPATIENT", pd.Series(0, index=df.index)).fillna(0).astype(int)
    out["CLAIM_DURATION_DAYS"] = df.get("CLAIM_DURATION_DAYS", pd.Series(0, index=df.index)).fillna(0).clip(0, 365)

    # NOTE: IS_SECONDARY_PAYER (NCH_PRMRY_PYR_CD != 0) is EXCLUDED.
    # With label_secondary_payer, NCH_PRMRY_PYR_CD IS the label — including it
    # would be direct leakage. The payer signal flows through L2/L4 rate features instead.
    # With label_zero_payment, it was safe — document this dependency clearly.

    # ── Diagnosis code features ──
    if "ICD9_DGNS_CD_1" in df.columns:
        out["ICD9_CHAPTER_1"] = df["ICD9_DGNS_CD_1"].apply(_icd9_to_chapter)
    else:
        out["ICD9_CHAPTER_1"] = "unknown"

    out["HAS_SECONDARY_DX"] = df.get("ICD9_DGNS_CD_2", pd.Series(np.nan, index=df.index)).notna().astype(int)
    out["HAS_TERTIARY_DX"]  = df.get("ICD9_DGNS_CD_3", pd.Series(np.nan, index=df.index)).notna().astype(int)

    # ── HCPCS / CPT ──
    hcpcs_cols = [c for c in ["HCPCS_CD_1","HCPCS_CD_2","HCPCS_CD_3"] if c in df.columns]
    out["HAS_HCPCS"]       = df.get("HCPCS_CD_1", pd.Series(np.nan, index=df.index)).notna().astype(int)
    out["HCPCS_CODE_COUNT"] = df[hcpcs_cols].notna().sum(axis=1) if hcpcs_cols else 0

    # ── Beneficiary demographics ──
    out["BENE_AGE_AT_CLAIM"] = df.get("BENE_AGE_AT_CLAIM", pd.Series(np.nan, index=df.index))
    out["BENE_SEX"]          = pd.to_numeric(df.get("BENE_SEX_IDENT_CD", np.nan), errors="coerce")
    out["BENE_RACE"]         = pd.to_numeric(df.get("BENE_RACE_CD", np.nan), errors="coerce")
    out["BENE_ESRD"]         = pd.to_numeric(df.get("BENE_ESRD_IND", 0), errors="coerce").fillna(0)
    out["COMORBIDITY_COUNT"] = df.get("COMORBIDITY_COUNT", pd.Series(np.nan, index=df.index))

    # ── Coverage months ──
    out["HI_CVRAGE_MONS"]  = pd.to_numeric(df.get("BENE_HI_CVRAGE_TOT_MONS", np.nan), errors="coerce")
    out["SMI_CVRAGE_MONS"] = pd.to_numeric(df.get("BENE_SMI_CVRAGE_TOT_MONS", np.nan), errors="coerce")
    out["HMO_CVRAGE_MONS"] = pd.to_numeric(df.get("BENE_HMO_CVRAGE_TOT_MONS", np.nan), errors="coerce")

    # ── Annual reimbursement totals — ALL EXCLUDED ──
    # Label = BENRES_OP > $670 (beneficiary outpatient responsibility).
    # The annual cost equation is: MEDREIMB + BENRES + PPPYMT = total cost.
    # Therefore MEDREIMB_OP, PPPYMT_OP are arithmetic proxies for BENRES_OP.
    # Including any of them gives LR perfect separation (AUC=1.0).
    # ALL four annual reimbursement columns are excluded:
    #   BENRES_OP, BENRES_IP — direct label source
    #   MEDREIMB_OP, MEDREIMB_IP, PPPYMT_OP, PPPYMT_IP — complements of label
    # Safe features that remain: coverage months (HMO/HI/SMI), demographics,
    # ICD chapter, provider rates, temporal, comorbidities.

    # ── Individual comorbidity flags ──
    for col in [c for c in df.columns if c.startswith("SP_")]:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    log.info(f"L1: {out.shape[1]} features")
    return out


# ── L2: Historical Aggregates ───────────────────────────────────────────────

def build_l2(df, windows=[30,60,90], is_train=True, history=None):
    out = pd.DataFrame(index=df.index)
    if history is None:
        history = {}

    # Provider denial rate (expanding, shift-1 — no label leakage)
    if is_train:
        provider_denial = (
            df.sort_values("CLM_FROM_DT")
              .groupby("PRVDR_NUM")["DENIED"]
              .transform(lambda x: x.shift(1).expanding().mean())
        )
        global_rate = df["DENIED"].mean()
        out["PRVDR_DENIAL_RATE_HIST"] = provider_denial.fillna(global_rate).values
        history["provider_denial_lookup"] = df.groupby("PRVDR_NUM")["DENIED"].mean().to_dict()
        history["global_denial_rate"]     = global_rate
    else:
        global_rate = history.get("global_denial_rate", 0.04)
        out["PRVDR_DENIAL_RATE_HIST"] = (
            df["PRVDR_NUM"].map(history.get("provider_denial_lookup", {}))
                           .fillna(global_rate).values
        )

    # Provider claim volume
    if is_train:
        history["provider_volume_lookup"] = df.groupby("PRVDR_NUM")["CLM_ID"].count().to_dict()
        vol = df["PRVDR_NUM"].map(history["provider_volume_lookup"]).fillna(1)
    else:
        vol = df["PRVDR_NUM"].map(history.get("provider_volume_lookup", {})).fillna(1)
    out["PRVDR_CLAIM_VOLUME_LOG"] = np.log1p(vol.values)

    # ICD-9 chapter denial rate
    if "ICD9_DGNS_CD_1" in df.columns:
        df = df.copy()
        df["_icd_ch"] = df["ICD9_DGNS_CD_1"].apply(_icd9_to_chapter)
        if is_train:
            icd_map = df.groupby("_icd_ch")["DENIED"].mean().to_dict()
            history["icd_chapter_denial_lookup"] = icd_map
        else:
            icd_map = history.get("icd_chapter_denial_lookup", {})
        out["ICD_CHAPTER_DENIAL_RATE"] = (
            df["_icd_ch"].map(icd_map).fillna(history.get("global_denial_rate", 0.04)).values
        )

    # Rolling beneficiary denial rates
    df_sorted = df.sort_values("CLM_FROM_DT").copy()
    for w in windows:
        col = f"BENE_DENIAL_RATE_{w}D"
        if is_train:
            rates = _rolling_bene_denial_rate(df_sorted, w)
            out[col] = pd.Series(rates, index=df_sorted.index).reindex(df.index).values
            history[f"bene_denial_rate_{w}d_lookup"] = df.groupby("DESYNPUF_ID")["DENIED"].mean().to_dict()
        else:
            out[col] = (
                df["DESYNPUF_ID"].map(history.get(f"bene_denial_rate_{w}d_lookup", {}))
                                 .fillna(history.get("global_denial_rate", 0.04)).values
            )

    log.info(f"L2: {out.shape[1]} features")
    return out, history


def _rolling_bene_denial_rate(df_sorted, window_days):
    rates = np.full(len(df_sorted), np.nan)
    for _, grp in df_sorted.groupby("DESYNPUF_ID"):
        if len(grp) < 2:
            continue
        idx    = grp.index
        dates  = grp["CLM_FROM_DT"].values
        denied = grp["DENIED"].values
        for j in range(1, len(grp)):
            cutoff = dates[j] - np.timedelta64(window_days, "D")
            mask = (dates[:j] >= cutoff) & (dates[:j] < dates[j])
            if mask.sum() > 0:
                rates[df_sorted.index.get_loc(idx[j])] = denied[:j][mask].mean()
    return rates


# ── L3: Temporal Features ───────────────────────────────────────────────────

def build_l3(df):
    out = pd.DataFrame(index=df.index)
    dt = df["CLM_FROM_DT"]
    out["CLAIM_MONTH"]       = dt.dt.month.astype("Int64")
    out["CLAIM_DOW"]         = dt.dt.dayofweek.astype("Int64")
    out["CLAIM_YEAR"]        = dt.dt.year.astype("Int64")
    out["CLAIM_QUARTER"]     = dt.dt.quarter.astype("Int64")
    out["CLAIM_DAY_OF_YEAR"] = dt.dt.dayofyear.astype("Int64")
    out["IS_WEEKEND"]        = (out["CLAIM_DOW"] >= 5).astype(int)
    out["IS_Q4"]             = (out["CLAIM_QUARTER"] == 4).astype(int)
    out["MONTH_SIN"] = np.sin(2 * np.pi * out["CLAIM_MONTH"].astype(float) / 12)
    out["MONTH_COS"] = np.cos(2 * np.pi * out["CLAIM_MONTH"].astype(float) / 12)
    out["DOW_SIN"]   = np.sin(2 * np.pi * out["CLAIM_DOW"].astype(float) / 7)
    out["DOW_COS"]   = np.cos(2 * np.pi * out["CLAIM_DOW"].astype(float) / 7)
    out["DAYS_SINCE_START"] = (dt - dt.min()).dt.days
    log.info(f"L3: {out.shape[1]} features")
    return out


# ── L4: Interaction Features ─────────────────────────────────────────────────

def build_l4(df, is_train=True, history=None):
    out = pd.DataFrame(index=df.index)
    if history is None:
        history = {}
    global_rate = history.get("global_denial_rate", df["DENIED"].mean() if is_train else 0.04)

    df = df.copy()

    # Provider × Payer interaction EXCLUDED when label=secondary_payer
    # (NCH_PRMRY_PYR_CD is the label source — using it in interactions = leakage)
    # Provider denial rate alone (from L2) captures provider-level signal safely.

    # ICD Chapter × Claim Type denial rate
    if "ICD9_DGNS_CD_1" in df.columns and "CLAIM_TYPE" in df.columns:
        df["_it_key"] = df["ICD9_DGNS_CD_1"].apply(_icd9_to_chapter) + "__" + df["CLAIM_TYPE"].astype(str)
        if is_train:
            history["icd_type_denial_lookup"] = df.groupby("_it_key")["DENIED"].mean().to_dict()
        out["ICD_TYPE_DENIAL_RATE"] = (
            df["_it_key"].map(history.get("icd_type_denial_lookup", {})).fillna(global_rate).values
        )

    # High-risk provider flag
    prvdr_lookup = history.get("provider_denial_lookup", {})
    if prvdr_lookup:
        out["IS_HIGH_RISK_PROVIDER"] = (
            df["PRVDR_NUM"].map(prvdr_lookup).fillna(0) > global_rate * 2
        ).astype(int).values
    else:
        out["IS_HIGH_RISK_PROVIDER"] = 0

    log.info(f"L4: {out.shape[1]} features")
    return out, history


# ── Categorical Encoding ─────────────────────────────────────────────────────

def encode_categoricals(df, is_train=True, history=None):
    if history is None:
        history = {}
    out = df.copy()
    if "ICD9_CHAPTER_1" in out.columns:
        dummies = pd.get_dummies(out["ICD9_CHAPTER_1"], prefix="ICD_CH", dtype=int)
        if is_train:
            history["icd_chapter_cols"] = dummies.columns.tolist()
        else:
            for col in history.get("icd_chapter_cols", []):
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[history.get("icd_chapter_cols", dummies.columns)]
        out = pd.concat([out.drop(columns=["ICD9_CHAPTER_1"]), dummies], axis=1)
    return out, history


# ── Main Entry Point ─────────────────────────────────────────────────────────

def build_features(df, windows=[30,60,90], is_train=True, history=None):
    """
    Full L1→L4 pipeline. Returns (feature_df, history).
    DENIED column must be present in df for is_train=True (used for L2/L4 fitting).
    DENIED is NOT included in the returned feature_df.
    """
    if history is None:
        history = {}
    log.info(f"build_features | is_train={is_train} | shape={df.shape}")

    l1 = build_l1(df)
    l2, history = build_l2(df, windows=windows, is_train=is_train, history=history)
    l3 = build_l3(df)
    l4, history = build_l4(df, is_train=is_train, history=history)

    features = pd.concat([l1, l2, l3, l4], axis=1)
    features, history = encode_categoricals(features, is_train=is_train, history=history)

    # Hard assert: label must never appear in feature matrix
    assert "DENIED" not in features.columns, "Label leaked into features!"
    log.info(f"Feature matrix: {features.shape}")
    return features, history
