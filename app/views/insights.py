"""Customer insights: which segments leave most, in business language."""
import plotly.express as px
import streamlit as st

from app.common import BLUE, RED, get_history
from src import config

df = get_history()
overall = df[config.TARGET].mean()


def rate(col):
    return df.groupby(col, observed=True)[config.TARGET].mean()


st.title("📊 Customer insights")
st.caption(f"Based on {len(df):,} past customers. Overall, {overall:.0%} of customers left the bank.")

# ---- Key findings (computed from the data, not hard-coded) -------------------
geo, act, prod, age, gen = rate("Geography"), rate("Activity"), rate("NumOfProducts"), rate("Age group"), rate("Gender")
top_geo, top_age = geo.idxmax(), age.idxmax()
others_geo = geo.drop(top_geo).mean()
multi = df[df.NumOfProducts >= 3][config.TARGET].mean()

st.subheader("Key findings")
cards = [
    ("🗺️", f"Customers in {top_geo}", f"leave at **{geo[top_geo]:.0%}**, about {geo[top_geo] / others_geo:.0f}× "
                                    f"the rate elsewhere ({others_geo:.0%})."),
    ("💤", "Inactive customers", f"leave at **{act['Inactive']:.0%}**, against {act['Active']:.0%} for active ones."),
    ("📦", "Customers with 3+ products", f"leave at **{multi:.0%}**. Customers with exactly 2 products are the most "
                                        f"loyal ({prod[2]:.0%} leave)."),
    ("🎂", f"Customers aged {top_age}", f"have the highest churn rate: **{age[top_age]:.0%}**."),
]
cols = st.columns(4)
for col, (icon, title, text) in zip(cols, cards):
    with col.container(border=True, height=160):
        st.markdown(f"#### {icon} {title}")
        st.markdown(text)

# Factors that barely matter, so teams don't waste effort on them.
weak = []
for col, label in [("Card Type", "card tier"), ("Satisfaction Score", "satisfaction score"),
                   ("Has credit card", "credit card ownership"), ("Tenure", "years with the bank")]:
    r = rate(col)
    if r.max() - r.min() < 0.05:
        weak.append(label)
if weak:
    st.caption(f"Little difference in churn by {', '.join(weak[:-1]) + ' or ' + weak[-1] if len(weak) > 1 else weak[0]}. "
               "These are not useful for targeting.")

# ---- Charts ---------------------------------------------------------------------
st.subheader("Churn rate by customer segment")
st.caption(f"The dashed line marks the overall rate ({overall:.0%}). Bars above it are higher-risk segments.")


def segment_chart(col, title, order=None):
    r = df.groupby(col, observed=True)[config.TARGET].agg(["mean", "size"]).reset_index()
    if order:
        r[col] = r[col].astype(str)
        r = r.set_index(col).reindex(order).reset_index()
    colors = [RED if m > overall else BLUE for m in r["mean"]]
    fig = px.bar(r, x=col, y="mean", text=r["mean"].map("{:.0%}".format),
                 custom_data=["size"], labels={"mean": "Churn rate", col: ""})
    fig.update_traces(marker_color=colors, hovertemplate="%{x}: %{y:.0%} of %{customdata[0]:,} customers left")
    fig.add_hline(y=overall, line_dash="dash", line_color="grey")
    fig.update_layout(title=title, height=300, margin=dict(t=40, b=10, l=10, r=10),
                      yaxis_tickformat=".0%", xaxis_type="category", yaxis_range=[0, min(1.15, r["mean"].max() * 1.25)])
    return fig


charts = [
    ("Geography", "By country", None),
    ("Products held", "By number of products", ["1 product", "2 products", "3 products", "4 products"]),
    ("Activity", "By account activity", ["Active", "Inactive"]),
    ("Age group", "By age group", ["18-29", "30-39", "40-49", "50-59", "60+"]),
    ("Gender", "By gender", None),
    ("Balance status", "By account balance", ["Zero balance", "Has a balance"]),
]
for i in range(0, len(charts), 2):
    c1, c2 = st.columns(2)
    for col, (feat, title, order) in zip((c1, c2), charts[i:i + 2]):
        with col.container(border=True):
            st.plotly_chart(segment_chart(feat, title, order), width="stretch")

st.caption("These figures describe past customers. To score current customers, use **Score a customer list**.")
