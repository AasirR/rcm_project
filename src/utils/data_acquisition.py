"""
src/utils/data_acquisition.py

Downloads and prepares CMS Medicare Synthetic Public Use Files (DE-SynPUF).

Usage:
    python -m src.utils.data_acquisition --samples 1 --out data/raw --save-parquet

CMS DE-SynPUF file docs:
https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf
"""

import argparse
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CMS DE-SynPUF download URLs — verified March 2026
# Source: https://www.cms.gov/.../de10-sample-1
# ---------------------------------------------------------------------------
CMS_BASE  = "https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads"
CMS_SITES = "https://www.cms.gov/sites/default/files/2020-09"

FILE_CONFIGS = {
    "beneficiary_2008": {
        "url_template": CMS_BASE  + "/de1_0_2008_beneficiary_summary_file_sample_{sample}.zip",
        "type": "beneficiary",
    },
    "beneficiary_2009": {
        "url_template": CMS_BASE  + "/de1_0_2009_beneficiary_summary_file_sample_{sample}.zip",
        "type": "beneficiary",
    },
    "beneficiary_2010": {
        "url_template": CMS_SITES + "/DE1_0_2010_Beneficiary_Summary_File_Sample_{sample}.zip",
        "type": "beneficiary",
    },
    "inpatient": {
        "url_template": CMS_BASE  + "/de1_0_2008_to_2010_inpatient_claims_sample_{sample}.zip",
        "type": "inpatient",
    },
    "outpatient": {
        "url_template": CMS_BASE  + "/de1_0_2008_to_2010_outpatient_claims_sample_{sample}.zip",
        "type": "outpatient",
    },
}

INPATIENT_COLS = [
    "DESYNPUF_ID","CLM_ID","SEGMENT","CLM_FROM_DT","CLM_THRU_DT",
    "PRVDR_NUM","AT_PHYSN_NPI","AT_PHYSN_UPIN","OP_PHYSN_NPI","OP_PHYSN_UPIN",
    "CLM_PMT_AMT","CLM_PASS_THRU_PER_DIEM_AMT","NCH_PRMRY_PYR_CLM_PD_AMT",
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM","CLM_UTLZTN_DAY_CNT","NCH_BENE_DSCHRG_DT",
    "CLM_DRG_CD","ICD9_DGNS_CD_1","ICD9_DGNS_CD_2","ICD9_DGNS_CD_3",
    "ICD9_PRCDR_CD_1","ICD9_PRCDR_CD_2","HCPCS_CD_1","HCPCS_CD_2","NCH_PRMRY_PYR_CD",
]

OUTPATIENT_COLS = [
    "DESYNPUF_ID","CLM_ID","SEGMENT","CLM_FROM_DT","CLM_THRU_DT",
    "PRVDR_NUM","AT_PHYSN_NPI","AT_PHYSN_UPIN","OP_PHYSN_NPI","OP_PHYSN_UPIN",
    "CLM_PMT_AMT","NCH_PRMRY_PYR_CLM_PD_AMT","NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
    "ICD9_DGNS_CD_1","ICD9_DGNS_CD_2","ICD9_DGNS_CD_3","ICD9_PRCDR_CD_1",
    "HCPCS_CD_1","HCPCS_CD_2","HCPCS_CD_3","NCH_PRMRY_PYR_CD","CLM_SRVC_CLSFCTN_TYPE_CD",
]

BENE_COLS = [
    "DESYNPUF_ID","BENE_BIRTH_DT","BENE_DEATH_DT","BENE_SEX_IDENT_CD","BENE_RACE_CD",
    "BENE_ESRD_IND","SP_STATE_CODE","BENE_COUNTY_CD","BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS","BENE_HMO_CVRAGE_TOT_MONS","PLAN_CVRG_MOS_NUM",
    "SP_ALZHDMTA","SP_CHF","SP_CHRNKIDN","SP_CNCR","SP_COPD","SP_DEPRESSN",
    "SP_DIABETES","SP_ISCHMCHT","SP_OSTEOPRS","SP_RA_OA","SP_STRKETIA",
    "MEDREIMB_IP","BENRES_IP","PPPYMT_IP","MEDREIMB_OP","BENRES_OP","PPPYMT_OP",
]


def download_file(url, dest, chunk_size=8192):
    if dest.exists():
        log.info(f"Already downloaded: {dest.name}")
        return dest
    log.info(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    log.info(f"Saved: {dest}")
    return dest


def extract_zip(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
        return [out_dir / name for name in zf.namelist()]


def load_csv(csv_path, cols, claim_type):
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    use_cols = [c for c in cols if c in available]
    df = pd.read_csv(csv_path, usecols=use_cols, dtype=str, low_memory=False)
    df["CLAIM_TYPE"] = claim_type
    return df


def acquire_sample(sample_id, out_dir):
    raw_dir = out_dir / f"sample_{sample_id:02d}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    claim_dfs = []
    bene_dfs  = []

    for file_key, config in FILE_CONFIGS.items():
        url   = config["url_template"].format(sample=sample_id)
        fname = url.split("/")[-1]
        ftype = config["type"]
        zip_dest = raw_dir / fname

        try:
            zip_path  = download_file(url, zip_dest)
            csvs      = extract_zip(zip_path, raw_dir)
            csv_files = [p for p in csvs if str(p).lower().endswith(".csv")]
            if not csv_files:
                log.warning(f"No CSV in {zip_path.name}")
                continue

            if ftype == "beneficiary":
                df = load_csv(csv_files[0], BENE_COLS, "beneficiary")
                log.info(f"Loaded {file_key}: {len(df):,} rows")
                bene_dfs.append(df)
            elif ftype == "inpatient":
                df = load_csv(csv_files[0], INPATIENT_COLS, "inpatient")
                log.info(f"Loaded inpatient: {len(df):,} rows")
                claim_dfs.append(df)
            elif ftype == "outpatient":
                df = load_csv(csv_files[0], OUTPATIENT_COLS, "outpatient")
                log.info(f"Loaded outpatient: {len(df):,} rows")
                claim_dfs.append(df)

        except requests.HTTPError as e:
            log.error(f"HTTP error {fname}: {e}")
        except Exception as e:
            log.error(f"Failed {fname}: {e}")

    if not claim_dfs:
        raise RuntimeError(f"No claim data loaded for sample {sample_id}")

    claims = pd.concat(claim_dfs, ignore_index=True)
    log.info(f"Total claims: {len(claims):,} rows")

    if bene_dfs:
        bene = (pd.concat(bene_dfs, ignore_index=True)
                  .drop(columns=["CLAIM_TYPE"], errors="ignore")
                  .drop_duplicates(subset=["DESYNPUF_ID"], keep="first"))
        claims = claims.merge(bene, on="DESYNPUF_ID", how="left")
        log.info(f"After bene merge: {len(claims):,} rows")

    return claims


def acquire_all(samples, out_dir):
    all_dfs = []
    for sid in samples:
        log.info(f"\n{'='*50}\nProcessing Sample {sid}\n{'='*50}")
        df = acquire_sample(sid, out_dir)
        df["SAMPLE_ID"] = sid
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Total combined rows: {len(combined):,}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Download CMS DE-SynPUF data")
    parser.add_argument("--samples", nargs="+", type=int, default=[1])
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--save-parquet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    df = acquire_all(args.samples, out_dir)

    if args.save_parquet:
        pq_path = out_dir / "cms_claims_combined.parquet"
        df.to_parquet(pq_path, index=False)
        log.info(f"Saved parquet: {pq_path}")
    else:
        csv_path = out_dir / "cms_claims_combined.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Saved CSV: {csv_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
