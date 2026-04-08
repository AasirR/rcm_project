"""
src/preprocessing/cleaner.py

Handles all raw data cleaning steps:
  - Drop inpatient-only columns that are >90% missing in the combined dataset
  - Parse dates
  - Cast financial columns to numeric
  - Derive claim duration
  - Standardise CLAIM_TYPE to a binary flag
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Columns that are >90% missing (inpatient-only fields).
# Confirmed from EDA: CLM_UTLZTN_DAY_CNT, CLM_PASS_THRU_PER_DIEM_AMT,
# NCH_BENE_DSCHRG_DT, CLM_DRG_CD are 92%+ missing.
# ICD9_PRCDR_CD_1/2 are 95-97% missing.
HIGH_MISSING_DROP = [
    "CLM_PASS_THRU_PER_DIEM_AMT",
    "NCH_BENE_DSCHRG_DT",
    "CLM_DRG_CD",
    "ICD9_PRCDR_CD_1",
    "ICD9_PRCDR_CD_2",
    "AT_PHYSN_UPIN",
    "OP_PHYSN_UPIN",
]

DATE_COLS = ["CLM_FROM_DT", "CLM_THRU_DT", "BENE_BIRTH_DT", "BENE_DEATH_DT"]

FINANCIAL_COLS = [
    "CLM_PMT_AMT",
    "NCH_PRMRY_PYR_CLM_PD_AMT",
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
    "MEDREIMB_IP",
    "BENRES_IP",
    "PPPYMT_IP",
    "MEDREIMB_OP",
    "BENRES_OP",
    "PPPYMT_OP",
]

NUMERIC_COLS = [
    "CLM_UTLZTN_DAY_CNT",
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "SP_STATE_CODE",
    "BENE_COUNTY_CD",
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA", "BENE_ESRD_IND",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a cleaned copy of df.
    Safe to call multiple times (idempotent).
    """
    df = df.copy()
    log.info(f"Input shape: {df.shape}")

    # 1. Drop high-missing inpatient-only columns
    drop_cols = [c for c in HIGH_MISSING_DROP if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    log.info(f"Dropped {len(drop_cols)} high-missing columns")

    # 2. Parse date columns (raw format: YYYYMMDD integer string)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

    # 3. Cast financial columns to float
    for col in FINANCIAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Cast other numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Claim duration (days)
    if "CLM_FROM_DT" in df.columns and "CLM_THRU_DT" in df.columns:
        df["CLAIM_DURATION_DAYS"] = (
            df["CLM_THRU_DT"] - df["CLM_FROM_DT"]
        ).dt.days.clip(lower=0)

    # 6. Binary flag: inpatient = 1, outpatient = 0
    if "CLAIM_TYPE" in df.columns:
        df["IS_INPATIENT"] = (df["CLAIM_TYPE"] == "inpatient").astype(int)

    # 7. Beneficiary age at claim date
    if "BENE_BIRTH_DT" in df.columns and "CLM_FROM_DT" in df.columns:
        df["BENE_AGE_AT_CLAIM"] = (
            (df["CLM_FROM_DT"] - df["BENE_BIRTH_DT"]).dt.days / 365.25
        ).round(1)
        # Sanity clip: Medicare population is generally 65+
        df["BENE_AGE_AT_CLAIM"] = df["BENE_AGE_AT_CLAIM"].clip(lower=0, upper=120)

    # 8. Comorbidity count (sum of SP_ flag columns)
    sp_cols = [c for c in df.columns if c.startswith("SP_") and c in df.columns]
    if sp_cols:
        # SP_ flags: 1 = has condition, 2 = does not have condition
        # Convert to binary: 1 → 1, 2 → 0
        for col in sp_cols:
            df[col] = np.where(df[col] == 1, 1, np.where(df[col] == 2, 0, np.nan))
        df["COMORBIDITY_COUNT"] = df[sp_cols].sum(axis=1, min_count=1)

    log.info(f"Output shape: {df.shape}")
    return df
