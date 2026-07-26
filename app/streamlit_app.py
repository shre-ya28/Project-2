"""
Streamlit dashboard — Buyer Segmentation & Investment Profiling
for Real Estate Market Intelligence (Parcl Co. Limited)

Run with:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Parcl | Buyer Segmentation Intelligence",
    page_icon="🏢",
    layout="wide",
)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    clustered_path = DATA_DIR / "clustered_clients.csv"
    summary_path = DATA_DIR / "cluster_summary.csv"

    if not clustered_path.exists():
        st.error(
            "Clustered data not found. Please run `python src/run_analysis.py` "
            "from the project root first to generate `data/clustered_clients.csv`."
        )
        st.stop()

    df = pd.read_csv(clustered_path)
    summary = pd.read_csv(summary_path)
    return df, summary


df, summary = load_data()

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.title("🏢 Parcl Buyer Intelligence")
st.sidebar.markdown("Filter the client base to explore behaviour by segment.")

countries = sorted(df["country"].dropna().unique().tolist())
regions = sorted(df["region"].dropna().unique().tolist())
purposes = sorted(df["acquisition_purpose"].dropna().unique().tolist())
client_types = sorted(df["client_type"].dropna().unique().tolist())
segments = sorted(df["segment_name"].dropna().unique().tolist())

sel_countries = st.sidebar.multiselect("Country", countries, default=[])
sel_regions = st.sidebar.multiselect("Region", regions, default=[])
sel_purpose = st.sidebar.multiselect("Acquisition Purpose", purposes, default=[])
sel_client_type = st.sidebar.multiselect("Client Type", client_types, default=[])
sel_segments = st.sidebar.multiselect("Buyer Segment", segments, default=[])

filtered = df.copy()
if sel_countries:
    filtered = filtered[filtered["country"].isin(sel_countries)]
if sel_regions:
    filtered = filtered[filtered["region"].isin(sel_regions)]
if sel_purpose:
    filtered = filtered[filtered["acquisition_purpose"].isin(sel_purpose)]
if sel_client_type:
    filtered = filtered[filtered["client_type"].isin(sel_client_type)]
if sel_segments:
    filtered = filtered[filtered["segment_name"].isin(sel_segments)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** buyer-clients")

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("Buyer Segmentation & Investment Profiling")
st.caption("Machine-learning based market intelligence for Parcl Co. Limited")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clients (filtered)", f"{len(filtered):,}")
k2.metric("Total Spend", f"${filtered['total_spend'].sum():,.0f}")
k3.metric("Avg Purchase Price", f"${filtered['avg_purchase_price'].mean():,.0f}" if len(filtered) else "—")
k4.metric("Avg Satisfaction", f"{filtered['satisfaction_score'].mean():.2f} / 5" if len(filtered) else "—")
k5.metric("Loan-Applied Rate", f"{filtered['loan_applied_flag'].mean()*100:.1f}%" if len(filtered) else "—")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Segmentation Overview", "💰 Investor Behaviour", "🌍 Geographic Analysis", "🔎 Segment Insights"]
)

# --------------------------------------------------------------------------
# TAB 1 — Buyer Segmentation Overview
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Cluster Distribution")
    col1, col2 = st.columns([1, 1.3])

    with col1:
        seg_counts = filtered["segment_name"].value_counts().reset_index()
        seg_counts.columns = ["segment_name", "count"]
        fig = px.pie(seg_counts, names="segment_name", values="count", hole=0.45,
                     title="Share of Clients by Segment")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            filtered, x="pca_1", y="pca_2", color="segment_name",
            hover_data=["client_id", "country", "total_spend", "avg_purchase_price"],
            title="Client Segments (PCA-reduced feature space)",
            labels={"pca_1": "Component 1", "pca_2": "Component 2"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Segment Definitions")
    st.dataframe(
        summary[["segment_name", "n_clients", "avg_age", "avg_spend", "avg_price",
                 "loan_rate", "avg_satisfaction"]]
        .rename(columns={
            "n_clients": "Clients", "avg_age": "Avg Age", "avg_spend": "Avg Total Spend",
            "avg_price": "Avg Purchase Price", "loan_rate": "Loan-Applied Rate",
            "avg_satisfaction": "Avg Satisfaction",
        })
        .style.format({
            "Avg Age": "{:.1f}", "Avg Total Spend": "${:,.0f}",
            "Avg Purchase Price": "${:,.0f}", "Loan-Applied Rate": "{:.1%}",
            "Avg Satisfaction": "{:.2f}",
        }),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# TAB 2 — Investor Behaviour Dashboard
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Investment Patterns by Segment")

    c1, c2 = st.columns(2)
    with c1:
        fig3 = px.bar(
            filtered.groupby("segment_name")["total_spend"].mean().reset_index(),
            x="segment_name", y="total_spend", color="segment_name",
            title="Average Total Spend by Segment", labels={"total_spend": "Avg Total Spend ($)"},
        )
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        fig4 = px.bar(
            filtered.groupby("segment_name")["avg_purchase_price"].mean().reset_index(),
            x="segment_name", y="avg_purchase_price", color="segment_name",
            title="Average Purchase Price by Segment", labels={"avg_purchase_price": "Avg Purchase Price ($)"},
        )
        st.plotly_chart(fig4, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig5 = px.bar(
            filtered.groupby("segment_name")["loan_applied_flag"].mean().reset_index(),
            x="segment_name", y="loan_applied_flag", color="segment_name",
            title="Loan-Applied Rate by Segment", labels={"loan_applied_flag": "Loan-Applied Rate"},
        )
        fig5.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig5, use_container_width=True)
    with c4:
        fig6 = px.box(
            filtered, x="segment_name", y="satisfaction_score", color="segment_name",
            title="Satisfaction Score Distribution by Segment",
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Acquisition Purpose & Financing")
    c5, c6 = st.columns(2)
    with c5:
        cross = pd.crosstab(filtered["segment_name"], filtered["acquisition_purpose"], normalize="index")
        fig7 = px.bar(cross, barmode="stack", title="Acquisition Purpose Mix by Segment")
        fig7.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig7, use_container_width=True)
    with c6:
        cross2 = pd.crosstab(filtered["segment_name"], filtered["referral_channel"], normalize="index")
        fig8 = px.bar(cross2, barmode="stack", title="Referral Channel Mix by Segment")
        fig8.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig8, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3 — Geographic Buyer Analysis
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Buyer Segments by Region")

    country_counts = filtered.groupby(["country", "segment_name"]).size().reset_index(name="count")
    fig9 = px.bar(
        country_counts, x="country", y="count", color="segment_name",
        title="Client Count by Country and Segment", barmode="stack",
    )
    st.plotly_chart(fig9, use_container_width=True)

    country_map_names = {"Usa": "United States", "Uk": "United Kingdom"}
    map_df = filtered.groupby("country")["total_spend"].sum().reset_index()
    map_df["country_display"] = map_df["country"].replace(country_map_names)
    fig10 = px.choropleth(
        map_df,
        locations="country_display", locationmode="country names", color="total_spend",
        title="Total Investment Spend by Country", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig10, use_container_width=True)

    st.subheader("Top Regions by Average Purchase Price")
    top_regions = (
        filtered.groupby("region")["avg_purchase_price"].mean()
        .sort_values(ascending=False).head(15).reset_index()
    )
    fig11 = px.bar(top_regions, x="avg_purchase_price", y="region", orientation="h",
                    title="Top 15 Regions by Average Purchase Price")
    st.plotly_chart(fig11, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 4 — Segment Insights Panel
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Descriptive Statistics per Segment")

    chosen = st.selectbox("Choose a segment to inspect", segments)
    seg_df = filtered[filtered["segment_name"] == chosen]

    if len(seg_df) == 0:
        st.warning("No clients in this segment for the current filters.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clients", f"{len(seg_df):,}")
        m2.metric("Avg Age", f"{seg_df['age'].mean():.1f}")
        m3.metric("Avg Total Spend", f"${seg_df['total_spend'].mean():,.0f}")
        m4.metric("Avg Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f} / 5")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top countries**")
            st.bar_chart(seg_df["country"].value_counts().head(10))
        with c2:
            st.markdown("**Client type split**")
            st.bar_chart(seg_df["client_type"].value_counts())

        st.markdown("**Raw client records (filtered)**")
        st.dataframe(
            seg_df[["client_id", "client_type", "gender", "age", "country", "region",
                    "acquisition_purpose", "loan_applied", "referral_channel",
                    "satisfaction_score", "num_purchases", "total_spend",
                    "avg_purchase_price"]],
            use_container_width=True,
        )

        csv = seg_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download this segment as CSV", csv,
                            file_name=f"{chosen.replace(' ', '_').lower()}_clients.csv",
                            mime="text/csv")

st.markdown("---")
st.caption(
    "Prototype dashboard for the *Machine Learning based Buyer Segmentation and "
    "Investment Profiling for Real Estate Market Intelligence* project · "
    "Unified Mentor × Parcl Co. Limited"
)
