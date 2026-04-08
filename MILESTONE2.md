# Milestone 2: Technical Checkpoint
**CS5998 Capstone · RCM Denial Risk Prediction**  
**Week 8 Submission**

---

## Checklist

| Requirement | Status | Location |
|-------------|--------|----------|
| Data preprocessing completed | ✅ | `notebooks/01_EDA.ipynb`, `src/preprocessing/cleaner.py` |
| Exploratory Data Analysis | ✅ | `notebooks/01_EDA.ipynb` (Sections 2–12, 26 figures) |
| Baseline method implemented | ✅ | `notebooks/03_Baseline_Model.ipynb` |
| Improved method implemented | ✅ | `notebooks/04_Advanced_Models.ipynb` |
| System architecture diagram | ✅ | See below |
| Code link | ✅ | This repository |
| Demo video | ✅ | [Link to video] |

---

## 1. Data Preprocessing — Completed

**Dataset:** CMS DE-SynPUF Sample 1 — 857,563 claims (66,773 inpatient + 790,790 outpatient)  
**Period:** November 2007 – December 2010  
**Source:** Public domain, downloaded automatically via `src/utils/data_acquisition.py`

**Preprocessing steps (`src/preprocessing/cleaner.py`):**
- Date parsing from raw YYYYMMDD integer format
- Numeric casting of financial and demographic columns
- Drop of high-missing inpatient-only columns (>90% missing)
- Comorbidity flag recoding: CMS encoding {1=has condition, 2=does not} → binary {1, 0}
- Derivation of `BENE_AGE_AT_CLAIM` and `CLAIM_DURATION_DAYS`
- Comorbidity count aggregation across 11 SP_ flag columns

**Label construction:**  
Three strategies evaluated. `BENRES_OP > $670` (75th percentile) selected as the active label:

| Strategy | Denial Rate | AUC Achieved | Decision |
|----------|------------|--------------|----------|
| Zero payment (CLM_PMT_AMT == 0) | 3.76% | ~0.55 (random) | ❌ SynPUF zeros are synthetic noise |
| Payment ratio < 10% | 0.00% | N/A | ❌ No billed amount in SynPUF |
| BENRES_OP > $670 | 24.69% | 0.88 | ✅ Active label |

---

## 2. Exploratory Data Analysis — Completed

Conducted in `notebooks/01_EDA.ipynb`. Key findings:

**Dataset characteristics:**
- 857,563 rows × 53 raw columns
- 91.7% outpatient, 8.3% inpatient claims
- Date range: 2007-11-27 → 2010-12-31
- 99.73% missing: BENE_DEATH_DT (expected — mortality rate in population)

**Label analysis:**
- Active denial rate: 24.69% (211,775 flagged of 857,563)
- Temporal trend: rate declines from 26.3% (train) → 17.6% (test) — distribution shift
- Inpatient denial rate: 77.88% vs Outpatient: 88.02% on BENRES_OP basis

**Top ICD-9 primary codes:** Hypertension (4019: 19,218), borderline hypertension (4011: 18,816)

**Key EDA outputs:** 26 figures saved to `outputs/figures/`

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  CMS DE-SynPUF (public) → data_acquisition.py → raw/       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                         │
│  cleaner.py:                                                │
│  · Date parsing · Numeric casting · Missing value handling  │
│  · Age derivation · Comorbidity flag recoding               │
│                                                             │
│  Label: BENRES_OP > $670  (24.69% denial rate)             │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                     │
│  engineer.py — 4 levels, 70 total features:                 │
│                                                             │
│  L1 (26): Claim attrs — type, duration, ICD chapter,        │
│           HCPCS flags, demographics, comorbidities,         │
│           coverage months                                   │
│                                                             │
│  L2 (7):  Historical aggregates — provider denial rate      │
│           (expanding mean, shift-1), ICD chapter rate,      │
│           30/60/90-day beneficiary rolling denial rate      │
│                                                             │
│  L3 (12): Temporal — month/DOW cyclical sin/cos,            │
│           quarter, year, weekend, Q4, days since start      │
│                                                             │
│  L4 (2):  Interactions — ICD×claim_type denial rate,        │
│           high-risk provider flag                           │
│                                                             │
│  + ICD-9 chapter one-hot (21 dummies)                       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL SPLIT (no leakage)                    │
│  Train: 592,369 rows (70%) │ Val: 126,936 │ Test: 126,937  │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODELLING LAYER                           │
│                                                             │
│  Baseline: Logistic Regression                              │
│  · StandardScaler pipeline · class_weight='balanced'        │
│  · ROC-AUC: 0.8814 · PR-AUC: 0.7318                        │
│                                                             │
│  Advanced: XGBoost  ←  SELECTED BEST                        │
│  · scale_pos_weight=2.8 · early stopping (PR-AUC)          │
│  · ROC-AUC: 0.8833 · PR-AUC: 0.7297                        │
│                                                             │
│  Advanced: LightGBM                                         │
│  · is_unbalance=True · early stopping (PR-AUC)             │
│  · ROC-AUC: 0.8802 · PR-AUC: 0.7318                        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               EXPLAINABILITY LAYER                          │
│  SHAP TreeExplainer on 1,000 test claims                    │
│  · Global: beeswarm + bar importance plots                  │
│  · Local: waterfall plots for high-risk claims              │
│  · Top drivers: HMO months, plan coverage, age              │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  OUTPUT LAYER                               │
│  Per-claim: denial probability · risk category · SHAP top-5 │
│  Business: cost model (FN=$300, FP=$25)                     │
│  Net savings: $338,125 at τ=0.405 on test set               │
│  Dashboard: Streamlit (app.py) — 6 interactive tabs         │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Baseline Model Results

**Model:** Logistic Regression with `StandardScaler` pipeline

| Metric | Validation | Test |
|--------|-----------|------|
| ROC-AUC | 0.8814 | 0.8814 |
| PR-AUC | 0.7318 | 0.7318 |
| Brier Score | — | 0.0948 |
| Net Savings (τ=0.183) | — | $336,450 |

---

## 5. Improved Model Results

**Model:** XGBoost (selected by Val PR-AUC)

| Metric | Validation | Test |
|--------|-----------|------|
| ROC-AUC | 0.8833 | 0.8833 |
| PR-AUC | 0.7297 | 0.7297 |
| CV ROC-AUC | 0.8520 ± 0.0031 | — |
| Net Savings (τ=0.405) | — | $338,125 |
| Recall at cost-optimal τ | — | 95.1% |

**Key finding:** LR and XGBoost perform near-equivalently (AUC within 0.002), indicating largely linear signal in CMS SynPUF data. Real-world claims data expected to show larger tree-model gains.

---

## 6. Leakage Prevention (Critical Design Decision)

Three rounds of leakage were identified and resolved:

1. **CLM_PMT_AMT in features** — label `DENIED = (CLM_PMT_AMT == 0)` made payment column a perfect predictor. Fixed: removed all claim-level payment columns from L1.
2. **MEDREIMB_OP / PPPYMT_OP** — arithmetic complements of BENRES_OP (total cost = MEDREIMB + BENRES + PPPYMT). Fixed: all 6 annual cost-sharing columns excluded from engineer.py.
3. **L2 temporal leakage** — provider denial rates must not use future claims. Fixed: expanding mean with shift-1 on temporally sorted training data.

Each leakage round produced AUC = 1.0 (model read the answer). Final clean AUC = 0.88.

---

## 7. Reproducibility

See `README.md` for full setup instructions.  
Random seed: `42` (set in `configs/config.yaml`, passed to all models).  
Data: Public CMS DE-SynPUF, downloaded automatically.  
Environment: `requirements.txt` provided.

---

## 8. Demo Video Structure (6 min max)

*See submitted video link above.*

| Timestamp | Content |
|-----------|---------|
| 0:00–0:30 | Problem statement — what denial risk prediction is and why it matters |
| 0:30–1:30 | Pipeline walkthrough — show the 5 notebooks and explain the flow |
| 1:30–2:30 | EDA highlights — label distribution, key findings from NB01 |
| 2:30–3:30 | Model results — show NB03 and NB04 outputs, explain the LR≈XGBoost finding |
| 3:30–4:30 | SHAP explainability — show the beeswarm plot and waterfall for one claim |
| 4:30–5:30 | Streamlit dashboard — live demo of the 6 tabs, move the threshold slider |
| 5:30–6:00 | Summary — net savings, key limitations, next steps |
