# RCM Predictive Analytics System
### Claim Denial Risk Prediction & Revenue Protection
**CS5998 Capstone Project — Master of Data Science & AI, Semester 3**

---

## Project Overview

An end-to-end predictive analytics system that estimates medical claim denial risk before submission, identifies explainable risk drivers via SHAP, and quantifies business impact through a cost-aware evaluation framework.

**Problem type:** Binary classification & risk scoring  
**Data type:** Tabular transactional (healthcare claims)  
**Techniques:** Feature engineering · Logistic Regression · XGBoost · LightGBM · SHAP  
**System context:** Batch scoring pipeline + interactive Streamlit dashboard  

---

## Results Summary

| Metric | Value |
|--------|-------|
| Best Model | XGBoost |
| Test ROC-AUC | 0.8833 |
| Test PR-AUC | 0.7297 |
| CV ROC-AUC (5-fold) | 0.8520 ± 0.0031 |
| Denials Caught (test) | 1,482 / 1,559 (95.1% recall) |
| Net Savings (test set) | $338,125 |
| Operating Threshold | τ = 0.405 |

---

## Project Structure

```
rcm_project/
├── configs/
│   └── config.yaml                  # Central config (paths, model params, thresholds)
├── data/
│   ├── raw/                         # Downloaded CMS DE-SynPUF files (git-ignored)
│   └── processed/                   # Feature sets & labelled data (git-ignored)
├── notebooks/
│   ├── 01_EDA.ipynb                 # Data acquisition & exploratory analysis
│   ├── 02_Feature_Engineering.ipynb # L1–L4 feature pipeline, temporal split
│   ├── 03_Baseline_Model.ipynb      # Logistic Regression baseline
│   ├── 04_Advanced_Models.ipynb     # XGBoost & LightGBM with early stopping
│   └── 05_Explainability_Evaluation.ipynb  # SHAP + business impact report
├── outputs/
│   ├── figures/                     # All saved plots (26 figures)
│   ├── models/                      # Serialised model artefacts
│   └── reports/                     # JSON reports + scored claims
├── src/
│   ├── preprocessing/
│   │   └── cleaner.py               # Data cleaning pipeline
│   ├── features/
│   │   └── engineer.py              # L1–L4 feature engineering
│   ├── evaluation/
│   │   └── metrics.py               # Shared evaluation utilities
│   └── utils/
│       ├── config_loader.py         # YAML config loader
│       └── data_acquisition.py      # CMS data download script
├── app.py                           # Streamlit interactive dashboard
├── requirements.txt                 # Python dependencies
└── MILESTONE2.md                    # Technical checkpoint document
```

---

## Quickstart

### 1. Prerequisites
- Python 3.12 (recommended — full wheel support for all ML packages)
- ~2GB disk space for CMS data

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/<your-username>/rcm_project.git
cd rcm_project

# Create virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Pipeline (in order)

```powershell
jupyter notebook
```

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| `01_EDA.ipynb` | Download CMS data + EDA | ~15 min (includes download) |
| `02_Feature_Engineering.ipynb` | Build feature matrix | ~10 min |
| `03_Baseline_Model.ipynb` | Train LR baseline | ~3 min |
| `04_Advanced_Models.ipynb` | Train XGBoost & LightGBM | ~15 min |
| `05_Explainability_Evaluation.ipynb` | SHAP + business report | ~5 min |

### 4. Launch the Dashboard

```powershell
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Data Source

**CMS 2008–2010 Data Entrepreneurs Synthetic Public Use File (DE-SynPUF)**  
- Public domain, no registration required  
- 857,563 anonymised Medicare claims (Sample 1 of 20)  
- Download handled automatically by `01_EDA.ipynb`  
- Docs: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files

**Label:** `BENRES_OP > $670` — beneficiary outpatient responsibility above the 75th percentile. Proxy for partial denial / coverage dispute. Causally linked to HMO coverage type, plan, diagnosis, and provider.

---

## Pipeline Architecture

```
Raw CMS Claims
      ↓
Preprocessing  (cleaner.py)
  · Date parsing · Numeric casting · Comorbidity recoding · Age derivation
      ↓
Label Construction  (NB01)
  · BENRES_OP > $670 · 24.69% denial rate
      ↓
Feature Engineering  (engineer.py)
  · L1: Claim attributes (26 features)
  · L2: Historical aggregates — provider/ICD denial rates, rolling windows (7)
  · L3: Temporal — cyclical encodings, seasonality (12)
  · L4: Interaction — ICD × claim type, high-risk provider flag (2)
  · ICD-9 chapter one-hot (21)
  · Total: 70 features
      ↓
Temporal Split  (70/15/15 by CLM_FROM_DT)
      ↓
Modelling
  · Baseline: Logistic Regression (class_weight='balanced')
  · Advanced: XGBoost + LightGBM (early stopping on val PR-AUC)
      ↓
Explainability  (SHAP TreeExplainer)
  · Global: feature importance + beeswarm
  · Local: waterfall plots for high-risk claims
      ↓
Risk Scoring Output
  · Per-claim: denial probability · risk category · top SHAP drivers
      ↓
Business Impact
  · Cost model: FN=$300 · FP=$25
  · Net savings: $338,125 on test set
```

---

## Key Design Decisions

**Why `BENRES_OP > $670` as label?**  
CMS SynPUF does not include an explicit denial flag. `CLM_PMT_AMT == 0` (zero payment) was tested first but achieved AUC ≈ 0.55 — SynPUF zeros are synthetically assigned with no clinical correlation. `BENRES_OP` (annual beneficiary outpatient responsibility) is causally determined by HMO plan type, diagnosis, provider, and state — all present in the feature matrix — and produces AUC 0.88.

**Why temporal split instead of random split?**  
Random splits allow future claims to inform past predictions. Temporal 70/15/15 split by `CLM_FROM_DT` prevents this, producing a realistic production scenario.

**Why PR-AUC as primary metric?**  
At 24.7% positive class prevalence, ROC-AUC can be misleadingly high for imbalanced classifiers. PR-AUC focuses on the minority (denied) class and is directly relevant to the clinical/business use case.

**Leakage prevention:**  
All annual cost-sharing columns (`BENRES_OP`, `BENRES_IP`, `MEDREIMB_*`, `PPPYMT_*`) are excluded from the feature matrix — they are arithmetic complements of the label. L2 historical features use expanding means with a one-period shift on sorted training data.

---

## Random Seeds

All random operations use seed `42`, set in `configs/config.yaml` and passed explicitly to all models and split functions.

---

## Hardware

Developed and tested on:
- **OS:** Windows 11
- **CPU:** Standard laptop (no GPU required)
- **RAM:** 16GB recommended (8GB minimum)
- **Python:** 3.12.x
- **Training time:** ~30 min end-to-end (NB01–NB05)

---

## Academic Integrity Note

This project was developed with AI assistance (Claude, Anthropic) for code scaffolding and debugging. All modelling decisions, label design rationale, leakage analysis, and business interpretation were developed and are understood by the author. The system architecture, feature engineering layers, and evaluation framework reflect original design choices made in response to the specific challenges of the CMS SynPUF dataset.
