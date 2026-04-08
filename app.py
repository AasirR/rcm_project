"""
RCM Denial Risk Dashboard
Run with: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RCM Denial Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #1F497D;
        margin-bottom: 8px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #666; font-weight: 500; }
    .metric-card h2 { margin: 4px 0 0 0; font-size: 28px; color: #1F497D; font-weight: 700; }
    .metric-card p  { margin: 2px 0 0 0; font-size: 12px; color: #888; }
    .risk-high   { background: #fff0f0; border-left-color: #c0392b; }
    .risk-high h2 { color: #c0392b; }
    .risk-medium { background: #fff8f0; border-left-color: #e67e22; }
    .risk-medium h2 { color: #e67e22; }
    .risk-low    { background: #f0f9f0; border-left-color: #27ae60; }
    .risk-low h2 { color: #27ae60; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    scored_path = ROOT / "outputs" / "reports" / "scored_test_claims.parquet"
    report_path = ROOT / "outputs" / "reports" / "final_report.json"
    cmp_path    = ROOT / "outputs" / "reports" / "model_comparison.json"

    if not scored_path.exists():
        return None, None, None

    scored = pd.read_parquet(scored_path)
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    cmp    = json.loads(cmp_path.read_text())    if cmp_path.exists()    else {}
    return scored, report, cmp

scored, report, cmp = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.title("RCM Denial Risk")
    st.caption("Predictive Analytics Dashboard")
    st.divider()

    if scored is None:
        st.error("No scored data found.\nRun Notebook 05 first to generate\noutputs/reports/scored_test_claims.parquet")
        st.stop()

    st.subheader("⚙️ Operating Threshold")
    threshold = st.slider(
        "Decision Threshold (τ)",
        min_value=0.01, max_value=0.99,
        value=float(cmp.get("best_threshold", 0.405)),
        step=0.01,
        help="Claims with denial probability ≥ τ are flagged for review"
    )

    st.subheader("🔍 Filter Claims")
    risk_filter = st.multiselect(
        "Risk Category",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    st.divider()
    st.caption("CMS DE-SynPUF Sample 1\nLabel: BENRES_OP > $670\nModel: XGBoost")

# ── Compute threshold-dependent metrics ──────────────────────────────────────
FN_COST = 300
FP_COST = 25

y_prob   = scored["DENIAL_PROB"].values
y_actual = scored["ACTUAL_DENIED"].values

y_pred = (y_prob >= threshold).astype(int)
tp = int(((y_pred == 1) & (y_actual == 1)).sum())
fp = int(((y_pred == 1) & (y_actual == 0)).sum())
fn = int(((y_pred == 0) & (y_actual == 1)).sum())
tn = int(((y_pred == 0) & (y_actual == 0)).sum())

precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
net_savings = int(y_actual.sum() * FN_COST - (fn * FN_COST + fp * FP_COST))
flagged_pct = (tp + fp) / len(y_actual) * 100

# Risk categories
LOW_T, HIGH_T = 0.20, 0.60
def risk_cat(p):
    if p >= HIGH_T: return "High"
    if p >= LOW_T:  return "Medium"
    return "Low"

scored["RISK_CAT_LIVE"]  = scored["DENIAL_PROB"].apply(risk_cat)
scored["FLAGGED"]        = (scored["DENIAL_PROB"] >= threshold).astype(int)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.title("🏥 RCM Denial Risk — Interactive Dashboard")
st.caption(
    f"CMS DE-SynPUF Sample 1 · {len(scored):,} test claims · "
    f"XGBoost · τ = {threshold:.3f}"
)

tabs = st.tabs([
    "📊 Overview",
    "🎯 Threshold Analysis",
    "🔬 Risk Stratification",
    "🧠 Feature Importance",
    "📋 Claim Explorer",
    "📈 Model Comparison",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Performance at Current Threshold")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Claims", f"{len(scored):,}")
    with c2:
        st.metric("Actual Denials", f"{int(y_actual.sum()):,}",
                  f"{y_actual.mean()*100:.1f}% rate")
    with c3:
        st.metric("Denials Caught", f"{tp:,}",
                  f"{recall*100:.1f}% recall")
    with c4:
        st.metric("False Alarms", f"{fp:,}",
                  f"{flagged_pct:.1f}% flagged")
    with c5:
        delta_color = "normal" if net_savings >= 0 else "inverse"
        st.metric("Net Savings", f"${net_savings:,.0f}", delta_color=delta_color)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Score Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scored.loc[scored["ACTUAL_DENIED"]==0, "DENIAL_PROB"],
            name="Accepted", nbinsx=60,
            marker_color="steelblue", opacity=0.65,
            histnorm="probability density",
        ))
        fig.add_trace(go.Histogram(
            x=scored.loc[scored["ACTUAL_DENIED"]==1, "DENIAL_PROB"],
            name="Denied", nbinsx=60,
            marker_color="crimson", opacity=0.75,
            histnorm="probability density",
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                      annotation_text=f"τ={threshold:.3f}", annotation_position="top right")
        fig.update_layout(
            barmode="overlay", height=350,
            xaxis_title="Denial Probability",
            yaxis_title="Density",
            legend=dict(x=0.75, y=0.95),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Confusion Matrix")
        cm_vals = np.array([[tn, fp], [fn, tp]])
        cm_text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                   [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_vals,
            x=["Pred: Accepted", "Pred: Denied"],
            y=["Actual: Accepted", "Actual: Denied"],
            text=cm_text, texttemplate="%{text}",
            colorscale=[[0,"#f0f8ff"],[1,"#1F497D"]],
            showscale=False,
            textfont=dict(size=16, color="black"),
        ))
        fig_cm.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Precision", f"{precision:.3f}")
    with col_b:
        st.metric("Recall", f"{recall:.3f}")
    with col_c:
        st.metric("F1 Score", f"{f1:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Business Cost Model — Threshold Sweep")

    @st.cache_data
    def compute_threshold_curve(y_prob, y_actual):
        thresholds = np.linspace(0.01, 0.99, 300)
        baseline   = y_actual.sum() * FN_COST
        rows = []
        for t in thresholds:
            yp  = (y_prob >= t).astype(int)
            tp_ = int(((yp==1)&(y_actual==1)).sum())
            fp_ = int(((yp==1)&(y_actual==0)).sum())
            fn_ = int(((yp==0)&(y_actual==1)).sum())
            prec = tp_/(tp_+fp_) if (tp_+fp_)>0 else 0
            rec  = tp_/(tp_+fn_) if (tp_+fn_)>0 else 0
            f1_  = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            rows.append({
                "threshold": t, "tp": tp_, "fp": fp_, "fn": fn_,
                "precision": prec, "recall": rec, "f1": f1_,
                "net_savings": baseline - (fn_*FN_COST + fp_*FP_COST),
                "flagged_pct": (tp_+fp_)/len(y_actual)*100,
            })
        return pd.DataFrame(rows)

    curve = compute_threshold_curve(y_prob, y_actual)

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("Net Savings vs Threshold",
                        "Precision / Recall / F1",
                        "Net Savings vs % Claims Flagged"))

    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["net_savings"]/1000,
        line=dict(color="green", width=2), name="Net Savings"), row=1, col=1)
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["precision"],
        line=dict(color="steelblue", width=2), name="Precision"), row=1, col=2)
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["recall"],
        line=dict(color="darkorange", width=2), name="Recall"), row=1, col=2)
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["f1"],
        line=dict(color="green", width=2.5), name="F1"), row=1, col=2)
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", row=1, col=2)

    fig.add_trace(go.Scatter(x=curve["flagged_pct"], y=curve["net_savings"]/1000,
        line=dict(color="steelblue", width=2), name="Savings"), row=1, col=3)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=3)

    fig.update_layout(height=400, showlegend=True,
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.update_yaxes(title_text="Net Savings ($000s)", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_yaxes(title_text="Net Savings ($000s)", row=1, col=3)
    fig.update_xaxes(title_text="Threshold τ", row=1, col=1)
    fig.update_xaxes(title_text="Threshold τ", row=1, col=2)
    fig.update_xaxes(title_text="% Claims Flagged", row=1, col=3)
    st.plotly_chart(fig, use_container_width=True)

    best_t = curve.loc[curve["net_savings"].idxmax()]
    st.info(
        f"**Optimal threshold: τ = {best_t['threshold']:.3f}** — "
        f"Net savings ${best_t['net_savings']:,.0f} · "
        f"Recall {best_t['recall']*100:.1f}% · "
        f"Precision {best_t['precision']:.3f} · "
        f"{best_t['flagged_pct']:.1f}% of claims flagged"
    )

    st.subheader("Threshold Comparison Table")
    compare_rows = []
    for label, t in [
        ("Cost-Optimal", float(best_t["threshold"])),
        ("Current (slider)", threshold),
        ("Balanced F1", float(curve.loc[curve["f1"].idxmax(), "threshold"])),
        ("High Recall (τ=0.04)", 0.04),
    ]:
        row = curve.iloc[(curve["threshold"] - t).abs().argsort().iloc[0]]
        compare_rows.append({
            "Scenario": label,
            "τ": f"{t:.3f}",
            "Precision": f"{float(row['precision']):.3f}",
            "Recall": f"{float(row['recall']):.3f}",
            "F1": f"{float(row['f1']):.3f}",
            "Flagged": f"{float(row['flagged_pct']):.1f}%",
            "Net Savings": f"${float(row['net_savings']):,.0f}",
        })
    st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: RISK STRATIFICATION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Risk Category Analysis")

    risk_summary = (
        scored.groupby("RISK_CAT_LIVE")
        .agg(
            N_Claims    = ("ACTUAL_DENIED", "count"),
            N_Denied    = ("ACTUAL_DENIED", "sum"),
            Denial_Rate = ("ACTUAL_DENIED", "mean"),
            Avg_Prob    = ("DENIAL_PROB",   "mean"),
            N_Flagged   = ("FLAGGED",       "sum"),
        )
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
        .reset_index()
        .rename(columns={"RISK_CAT_LIVE": "Risk"})
    )

    col1, col2, col3 = st.columns(3)
    for col, cat, css in zip([col1, col2, col3],
                              ["High", "Medium", "Low"],
                              ["risk-high", "risk-medium", "risk-low"]):
        row = risk_summary[risk_summary["Risk"] == cat]
        if len(row):
            r = row.iloc[0]
            with col:
                st.markdown(f"""
                <div class="metric-card {css}">
                    <h3>{cat} Risk (≥{HIGH_T if cat=='High' else LOW_T if cat=='Medium' else 0:.0%})</h3>
                    <h2>{int(r['N_Claims']):,}</h2>
                    <p>claims · {r['Denial_Rate']*100:.1f}% actual denial rate</p>
                    <p>Avg probability: {r['Avg_Prob']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            risk_summary, x="Risk", y="N_Claims",
            color="Risk",
            color_discrete_map={"High":"crimson","Medium":"darkorange","Low":"steelblue"},
            text="N_Claims",
            title="Claim Volume by Risk Category",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, height=380,
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = px.bar(
            risk_summary, x="Risk", y="Denial_Rate",
            color="Risk",
            color_discrete_map={"High":"crimson","Medium":"darkorange","Low":"steelblue"},
            text=risk_summary["Denial_Rate"].map(lambda x: f"{x:.1%}"),
            title="Actual Denial Rate by Risk Category",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, height=380,
                           yaxis_tickformat=".0%",
                           margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Risk Distribution Breakdown")
    fig3 = px.histogram(
        scored, x="DENIAL_PROB", color="RISK_CAT_LIVE",
        nbins=80, barmode="stack",
        color_discrete_map={"High":"crimson","Medium":"darkorange","Low":"steelblue"},
        category_orders={"RISK_CAT_LIVE": ["High","Medium","Low"]},
        title="Score Distribution Coloured by Risk Category",
        labels={"DENIAL_PROB": "Denial Probability", "RISK_CAT_LIVE": "Risk"},
    )
    fig3.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Top Denial Risk Drivers")

    top_features = report.get("top_features", [])
    if not top_features:
        st.warning("Run Notebook 05 to generate SHAP feature importance.")
    else:
        # Build description table
        feat_descriptions = {
            "HMO_CVRAGE_MONS":        "HMO Coverage Months — more coverage = lower risk",
            "PLAN_CVRG_MOS_NUM":      "Part D Plan Coverage Months — longer = lower risk",
            "BENE_AGE_AT_CLAIM":      "Beneficiary Age — older patients face higher cost-sharing",
            "PRVDR_DENIAL_RATE_HIST": "Provider Historical Denial Rate — past billing patterns",
            "ICD_CHAPTER_DENIAL_RATE":"ICD-9 Chapter Denial Rate — diagnosis category risk",
            "COMORBIDITY_COUNT":      "Number of Chronic Conditions — higher = complex coverage",
            "CLAIM_DURATION_DAYS":    "Claim Duration — longer stays → higher cost exposure",
            "SMI_CVRAGE_MONS":        "Supplementary Medical Insurance Coverage Months",
            "HI_CVRAGE_MONS":         "Hospital Insurance Coverage Months",
            "IS_INPATIENT":           "Inpatient vs Outpatient — inpatient claims differ in cost-sharing",
        }

        feat_df = pd.DataFrame([
            {"Rank": i+1, "Feature": f,
             "Description": feat_descriptions.get(f, f),
             "Direction": "↓ Decreases Risk" if "CVRAGE" in f or "PLAN" in f
                         else "↑ Increases Risk" if "AGE" in f or "DENIAL" in f
                         else "Varies"}
            for i, f in enumerate(top_features[:10])
        ])
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("SHAP Insights")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
**🔵 Decreases Denial Risk (Protective Factors):**
- High HMO coverage months → plan caps cost-sharing
- Long Part D plan coverage → comprehensive benefits
- Low historical provider denial rate → clean billing history
- Supplementary/V-code diagnosis chapters

**🔴 Increases Denial Risk:**
- Few or zero HMO coverage months
- Advanced beneficiary age
- High-denial-rate providers
- Circulatory, musculoskeletal ICD chapters
            """)

        with col_b:
            st.markdown("""
**💡 Actionable Interventions:**

1. **Zero HMO months flagged** → verify primary payer and plan type before submission

2. **High-denial-rate provider** → trigger pre-submission review for all claims above τ

3. **Circulatory/musculoskeletal ICD** → cross-check diagnosis docs against payer coverage rules

4. **Elderly beneficiaries + complex comorbidities** → route to senior coder for plan-specific review

**📊 Model Finding:**
LR ≈ XGBoost (AUC within 0.002) — signal is largely linear. 
Real-world data expected to show larger tree-model gains.
            """)

    st.divider()
    st.subheader("Feature Distribution — Denied vs Accepted")
    available_cols = [c for c in scored.columns
                      if c not in ["DENIAL_PROB","LR_PROB","RISK_CATEGORY",
                                   "RISK_CAT_LIVE","FLAGGED","ACTUAL_DENIED","MODEL_FLAG"]
                      and scored[c].dtype in [np.float64, np.int64, float, int]]

    if available_cols:
        sel_feat = st.selectbox("Select feature to compare:", available_cols)
        fig_dist = go.Figure()
        for label, val, color in [("Accepted", 0, "steelblue"), ("Denied", 1, "crimson")]:
            vals = scored.loc[scored["ACTUAL_DENIED"]==val, sel_feat].dropna()
            fig_dist.add_trace(go.Histogram(
                x=vals, name=label, nbinsx=50,
                marker_color=color, opacity=0.65,
                histnorm="probability density",
            ))
        fig_dist.update_layout(
            barmode="overlay", height=320,
            xaxis_title=sel_feat, yaxis_title="Density",
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: CLAIM EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Individual Claim Explorer")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_sel = st.multiselect("Risk Category", ["High","Medium","Low"],
                                  default=risk_filter)
    with col_f2:
        flag_sel = st.radio("Flag Status", ["All","Flagged only","Not flagged"], horizontal=True)
    with col_f3:
        actual_sel = st.radio("Actual Denial", ["All","Denied","Accepted"], horizontal=True)

    filtered = scored.copy()
    if risk_sel:
        filtered = filtered[filtered["RISK_CAT_LIVE"].isin(risk_sel)]
    if flag_sel == "Flagged only":
        filtered = filtered[filtered["FLAGGED"] == 1]
    elif flag_sel == "Not flagged":
        filtered = filtered[filtered["FLAGGED"] == 0]
    if actual_sel == "Denied":
        filtered = filtered[filtered["ACTUAL_DENIED"] == 1]
    elif actual_sel == "Accepted":
        filtered = filtered[filtered["ACTUAL_DENIED"] == 0]

    st.caption(f"Showing {len(filtered):,} claims (of {len(scored):,} total)")

    display_cols = ["DENIAL_PROB","RISK_CAT_LIVE","FLAGGED","ACTUAL_DENIED"] + \
                   [c for c in scored.columns
                    if c not in ["DENIAL_PROB","LR_PROB","RISK_CATEGORY",
                                 "RISK_CAT_LIVE","FLAGGED","ACTUAL_DENIED","MODEL_FLAG"]]

    show_df = (filtered[display_cols]
               .sort_values("DENIAL_PROB", ascending=False)
               .head(500)
               .reset_index(drop=True))

    show_df["DENIAL_PROB"] = show_df["DENIAL_PROB"].map("{:.3f}".format)
    show_df = show_df.rename(columns={
        "DENIAL_PROB":    "Prob",
        "RISK_CAT_LIVE":  "Risk",
        "FLAGGED":        "Flagged",
        "ACTUAL_DENIED":  "Actual",
    })

    st.dataframe(
        show_df,
        hide_index=True,
        use_container_width=True,
        height=400,
    )

    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download filtered claims as CSV",
        csv, "filtered_claims.csv", "text/csv",
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Model Performance Comparison")

    if not cmp or "models" not in cmp:
        st.warning("Run Notebook 04 to generate model_comparison.json")
    else:
        models_df = pd.DataFrame(cmp["models"])

        col1, col2 = st.columns(2)
        with col1:
            fig_roc = px.bar(
                models_df, x="Model", y="Test ROC-AUC",
                color="Model",
                color_discrete_sequence=["#95a5a6","#1F497D","#e67e22"],
                title="Test ROC-AUC by Model",
                text="Test ROC-AUC",
            )
            fig_roc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_roc.update_layout(showlegend=False, height=380,
                                  yaxis=dict(range=[0.80, 0.92]),
                                  margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            fig_pr = px.bar(
                models_df, x="Model", y="Test PR-AUC",
                color="Model",
                color_discrete_sequence=["#95a5a6","#1F497D","#e67e22"],
                title="Test PR-AUC by Model",
                text="Test PR-AUC",
            )
            fig_pr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_pr.update_layout(showlegend=False, height=380,
                                 yaxis=dict(range=[0.60, 0.80]),
                                 margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_pr, use_container_width=True)

        st.subheader("Detailed Metrics Table")
        st.dataframe(models_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Cross-Validation Results (XGBoost)")
        cv = cmp.get("cv", {})
        if cv:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("CV ROC-AUC (mean)", f"{cv.get('roc_auc_mean',0):.4f}",
                          f"± {cv.get('roc_auc_std',0):.4f}")
            with col_b:
                st.metric("CV PR-AUC (mean)", f"{cv.get('pr_auc_mean',0):.4f}",
                          f"± {cv.get('pr_auc_std',0):.4f}")

        st.subheader("Key Finding")
        st.info("""
**LR ≈ XGBoost ≈ LightGBM** — all three models achieve ROC-AUC within 0.003 of each other.

This indicates the predictive signal in CMS SynPUF is **largely linear**: HMO coverage months, 
plan type, and beneficiary demographics correlate linearly with beneficiary cost-sharing responsibility.

Tree models provide marginal additional benefit here. On real-world claims data, non-linear 
payer-rule × diagnosis interactions are expected to give XGBoost a larger advantage over LR.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "RCM Denial Risk Dashboard · CMS DE-SynPUF Sample 1 · "
    "XGBoost (ROC-AUC 0.8833, PR-AUC 0.7297) · "
    "Label: BENRES_OP > $670 · MDS-AI Semester 3 Project"
)
