"""About the model: plain summary first, technical detail in expanders."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import BLUE, RED, get_metrics, ratio_phrase, require_artifacts, trust_statement
from src import config

art = require_artifacts()
meta, metrics = art.metadata, get_metrics()
t = art.threshold

st.title("ℹ️ About the model")

# ---- Plain-language summary ------------------------------------------------------
st.subheader("In short")
with st.container(border=True):
    st.markdown(
        f"""
- **What it does:** estimates how likely each customer is to close their account, based on their profile,
  products and activity.
- **How reliable it is:** {trust_statement(metrics)}
- **How risk levels are set:** a missed leaver is treated as **{meta['cost_fn'] / meta['cost_fp']:.0f}× more costly**
  than an unnecessary retention offer, so the tool flags customers early.
  - **High risk:** likelihood of {t:.0%} or more.
  - **Medium risk:** {t / 2:.0%} to {t:.0%}.
  - **Low risk:** below {t / 2:.0%}.
- **What it learned from:** {meta['train_rows']:,} past customers. It was then tested on
  {metrics['test_rows']:,} different customers it had never seen.
- **Last updated:** {pd.Timestamp(meta['trained_at']):%d %B %Y}.
"""
        if metrics else "Performance figures are not available yet."
    )
if metrics:
    saving = metrics["cost_saving_vs_no_model"]
    st.success(f"On the test customers, acting on this tool's flags would have cut the cost of lost customers "
               f"and wasted offers by about **{saving:.0%}** compared with doing nothing.", icon="📉")

st.subheader("Technical details")
st.caption("For data and risk teams.")

if metrics:
    m = metrics["metrics"]
    with st.expander("Performance on the test set", expanded=False):
        k = st.columns(5)
        k[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}", help="Ranking quality: 0.5 = random, 1.0 = perfect")
        k[1].metric("PR-AUC", f"{m['pr_auc']:.3f}", help=f"Random baseline = {metrics['test_churn_rate']:.3f}")
        k[2].metric("Recall", f"{m['recall']:.1%}", help="Share of actual leavers flagged High risk")
        k[3].metric("Precision", f"{m['precision']:.1%}", help="Share of High-risk customers who actually left")
        k[4].metric("Brier score", f"{m['brier']:.3f}", help="Probability accuracy: lower is better")
        crit = metrics["success_criteria"]
        st.markdown(" · ".join(f"{'✅' if c['met'] else '❌'} {n.upper().replace('_', '-')} target {c['target']} "
                               f"(actual {c['actual']:.3f})" for n, c in crit.items()))

        c1, c2, c3 = st.columns(3)
        cm = m["confusion_matrix"]
        fig = px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True,
                        x=["Predicted stay", "Predicted leave"], y=["Actually stayed", "Actually left"],
                        color_continuous_scale="Blues")
        fig.update_layout(title=f"Confusion matrix @ {t:.2f}", coloraxis_showscale=False, height=320,
                          margin=dict(t=40, b=10))
        c1.plotly_chart(fig, width="stretch")

        roc = metrics["curves"]["roc"]
        fig = go.Figure([go.Scatter(x=roc["fpr"], y=roc["tpr"], line_color=RED),
                         go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"))])
        fig.update_layout(title=f"ROC curve (AUC {m['roc_auc']:.3f})", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=320, margin=dict(t=40, b=10), showlegend=False)
        c2.plotly_chart(fig, width="stretch")

        cal = metrics["curves"]["calibration"]
        fig = go.Figure([go.Scatter(x=cal["mean_predicted"], y=cal["actual_rate"], mode="lines+markers",
                                    line_color=RED),
                         go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"))])
        fig.update_layout(title="Calibration", xaxis_title="Predicted probability", yaxis_title="Actual rate",
                          height=320, margin=dict(t=40, b=10), showlegend=False)
        c3.plotly_chart(fig, width="stretch")

    with st.expander("What drives the predictions"):
        imp = pd.Series(metrics["shap_importance"]).sort_values()
        fig = px.bar(imp, orientation="h", labels={"value": "Mean |SHAP value|", "index": ""})
        fig.update_traces(marker_color=BLUE)
        fig.update_layout(showlegend=False, height=420, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("SHAP values from the uncalibrated model score (log-odds). Calibration rescales the score "
                   "without changing the order of reasons.")

    with st.expander("Fairness check"):
        st.markdown("Recall and false-positive rate should be broadly similar across groups. "
                    "Large gaps mean some groups are flagged more or less often.")
        fair = pd.DataFrame(metrics["fairness"])
        attr = st.radio("Group by", config.FAIRNESS_COLS, horizontal=True)
        f = fair[fair.attribute == attr]
        long = f.melt(id_vars="group", value_vars=["actual_churn_rate", "flagged_rate", "recall",
                                                   "false_positive_rate"])
        fig = px.bar(long, x="group", y="value", color="variable", barmode="group",
                     labels={"value": "Rate", "group": attr, "variable": ""})
        fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(f.drop(columns="attribute").round(3), width="stretch", hide_index=True)
        st.caption("Gender is used as an input because it relates to churn in the data, but the tool never "
                   "recommends an action based on gender.")

with st.expander("Model and training"):
    st.markdown(
        f"- **Algorithm:** {meta['model_name']} ({meta['feature_set']} features), "
        f"{'tuned with Optuna' if meta['params'] else 'default settings'}, {meta['calibration']} calibration\n"
        f"- **Decision threshold:** {t:.2f}, chosen to minimise expected cost "
        f"(missed leaver = {meta['cost_fn']:.0f}, unnecessary offer = {meta['cost_fp']:.0f})\n"
        f"- **Training data:** {meta['train_rows']:,} customers, churn rate {meta['train_churn_rate']:.1%}\n"
        f"- **Trained:** {meta['trained_at'].replace('T', ' ')}"
    )
    comp = pd.DataFrame(meta["comparison"]).rename(columns={"cv_roc_auc": "CV ROC-AUC"})
    comp["CV ROC-AUC"] = comp["CV ROC-AUC"].round(4)
    st.markdown("**Models compared** (5-fold cross-validation on training data)")
    st.dataframe(comp, width="stretch", hide_index=True)
    st.json(meta["params"], expanded=False)

with st.expander("Data decisions"):
    st.markdown(
        f"- **`Complain` is excluded (data leakage).** It matches the churn outcome for 99.9% of customers. "
        "A complaint is most likely recorded when the customer leaves, so it would not be known early enough to act on.\n"
        f"- **Identifiers dropped:** {', '.join(f'`{c}`' for c in config.ID_COLS)}\n"
        f"- **Inputs used:** {', '.join(f'`{c}`' for c in meta['feature_columns'])}\n"
        "- **Limitations:** one snapshot of about 10,000 customers with no time dimension; the cost ratio is an "
        "assumption; no drift monitoring yet."
    )

with st.expander("Software versions"):
    st.json(meta["versions"], expanded=True)
