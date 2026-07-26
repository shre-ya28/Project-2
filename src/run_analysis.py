"""
run_analysis.py
----------------
End-to-end script: cleans data, engineers features, evaluates & fits
clustering models, saves the labelled dataset + all figures used in the
research paper and Streamlit app.

Usage:
    python src/run_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from data_processing import get_processed_data
from clustering import run_full_pipeline

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "data"

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = "deep"


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def eda_plots(clients, properties, features):
    # 1. Client type / gender / acquisition purpose distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    clients["client_type"].value_counts().plot(kind="bar", ax=axes[0], color=sns.color_palette(PALETTE))
    axes[0].set_title("Client Type")
    clients["gender"].value_counts().plot(kind="bar", ax=axes[1], color=sns.color_palette(PALETTE))
    axes[1].set_title("Gender")
    clients["acquisition_purpose"].value_counts().plot(kind="bar", ax=axes[2], color=sns.color_palette(PALETTE))
    axes[2].set_title("Acquisition Purpose")
    savefig("01_client_distributions.png")

    # 2. Top countries
    plt.figure(figsize=(9, 4.5))
    clients["country"].value_counts().plot(kind="bar", color=sns.color_palette(PALETTE))
    plt.title("Clients by Country")
    plt.ylabel("Number of clients")
    savefig("02_clients_by_country.png")

    # 3. Age distribution
    plt.figure(figsize=(8, 4.5))
    sns.histplot(clients["age"], bins=30, kde=True, color=sns.color_palette(PALETTE)[0])
    plt.title("Client Age Distribution")
    plt.xlabel("Age (years)")
    savefig("03_age_distribution.png")

    # 4. Satisfaction score
    plt.figure(figsize=(7, 4.5))
    sns.countplot(x="satisfaction_score", data=clients, color=sns.color_palette(PALETTE)[1])
    plt.title("Satisfaction Score Distribution")
    savefig("04_satisfaction_distribution.png")

    # 5. Loan applied vs acquisition purpose
    plt.figure(figsize=(7, 4.5))
    sns.countplot(x="acquisition_purpose", hue="loan_applied", data=clients)
    plt.title("Loan Applied vs Acquisition Purpose")
    savefig("05_loan_vs_purpose.png")

    # 6. Referral channel
    plt.figure(figsize=(7, 4.5))
    clients["referral_channel"].value_counts().plot(kind="bar", color=sns.color_palette(PALETTE))
    plt.title("Referral Channel")
    savefig("06_referral_channel.png")

    # 7. Sale price distribution + by unit category
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(properties["sale_price"], bins=40, ax=axes[0], color=sns.color_palette(PALETTE)[2])
    axes[0].set_title("Sale Price Distribution")
    sns.boxplot(x="unit_category", y="sale_price", data=properties, ax=axes[1])
    axes[1].set_title("Sale Price by Unit Category")
    savefig("07_price_distributions.png")

    # 8. Transactions over time
    ts = properties.dropna(subset=["transaction_date"]).copy()
    ts["month"] = ts["transaction_date"].dt.to_period("M").dt.to_timestamp()
    monthly = ts.groupby("month").size()
    plt.figure(figsize=(10, 4.5))
    monthly.plot(marker="o")
    plt.title("Transactions Over Time")
    plt.ylabel("Number of transactions")
    savefig("08_transactions_over_time.png")

    # 9. Correlation heatmap of numeric client features
    plt.figure(figsize=(7, 6))
    corr_cols = ["age", "satisfaction_score", "loan_applied_flag", "num_purchases",
                 "total_spend", "avg_purchase_price", "total_floor_area"]
    sns.heatmap(features[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlation Heatmap (numeric features)")
    savefig("09_correlation_heatmap.png")


def clustering_plots(result):
    eval_df = result["eval_df"]
    df = result["df"]

    # 10. Elbow method
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eval_df["k"], eval_df["inertia"], marker="o", color="tab:blue")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(eval_df["k"], eval_df["silhouette"], marker="s", color="tab:orange")
    ax2.set_ylabel("Silhouette Score", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("Elbow Method & Silhouette Score vs k")
    savefig("10_elbow_silhouette.png")

    # 11. PCA scatter colored by cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="pca_1", y="pca_2", hue="segment_name", data=df, palette=PALETTE, s=35, alpha=0.75)
    plt.title("Client Segments (PCA-reduced feature space)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("11_pca_clusters.png")

    # 12. Dendrogram (hierarchical clustering)
    from scipy.cluster.hierarchy import dendrogram, linkage
    from clustering import prepare_matrix
    sample = df.sample(min(150, len(df)), random_state=42)
    Z = linkage(sample[["pca_1", "pca_2"]], method="ward")
    plt.figure(figsize=(11, 5))
    dendrogram(Z, no_labels=True, color_threshold=0)
    plt.title("Hierarchical Clustering Dendrogram (sample of 150 clients, PCA space)")
    plt.xlabel("Clients")
    plt.ylabel("Distance")
    savefig("12_dendrogram.png")

    # 13. Segment sizes
    plt.figure(figsize=(7, 4.5))
    df["segment_name"].value_counts().plot(kind="bar", color=sns.color_palette(PALETTE))
    plt.title("Buyer Segment Sizes")
    plt.ylabel("Number of clients")
    savefig("13_segment_sizes.png")

    # 14. Avg spend / avg price by segment
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    df.groupby("segment_name")["total_spend"].mean().sort_values().plot(kind="barh", ax=axes[0], color=sns.color_palette(PALETTE))
    axes[0].set_title("Average Total Spend by Segment")
    df.groupby("segment_name")["avg_purchase_price"].mean().sort_values().plot(kind="barh", ax=axes[1], color=sns.color_palette(PALETTE))
    axes[1].set_title("Average Purchase Price by Segment")
    savefig("14_spend_price_by_segment.png")

    # 15. Loan rate & satisfaction by segment
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    df.groupby("segment_name")["loan_applied_flag"].mean().sort_values().plot(kind="barh", ax=axes[0], color=sns.color_palette(PALETTE))
    axes[0].set_title("Loan-Applied Rate by Segment")
    df.groupby("segment_name")["satisfaction_score"].mean().sort_values().plot(kind="barh", ax=axes[1], color=sns.color_palette(PALETTE))
    axes[1].set_title("Average Satisfaction Score by Segment")
    savefig("15_loan_satisfaction_by_segment.png")

    # 16. Geographic distribution by segment (top countries)
    plt.figure(figsize=(10, 5))
    top_countries = df["country"].value_counts().head(6).index
    sub = df[df["country"].isin(top_countries)]
    ct = pd.crosstab(sub["country"], sub["segment_name"], normalize="index")
    ct.plot(kind="bar", stacked=True, colormap="tab20")
    plt.title("Segment Composition by Country (top 6 countries)")
    plt.ylabel("Share of clients")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("16_segment_by_country.png")


def main():
    print("Loading & processing data...")
    clients, properties, features = get_processed_data()

    print("Running EDA plots...")
    eda_plots(clients, properties, features)

    print("Running clustering pipeline...")
    result = run_full_pipeline(features, k=4)

    print("Running clustering plots...")
    clustering_plots(result)

    # Save labelled dataset for the Streamlit app
    df = result["df"]
    df.to_csv(OUT_DIR / "clustered_clients.csv", index=False)

    # Save cluster summary table
    summary = result["summary"].copy()
    summary["segment_name"] = summary.index.map(result["segment_names"])
    summary.to_csv(OUT_DIR / "cluster_summary.csv")

    # Save evaluation table
    result["eval_df"].to_csv(OUT_DIR / "k_evaluation.csv", index=False)

    print("\nSegment names:", result["segment_names"])
    print("\nCluster summary:\n", summary)
    print("\nAdjusted Rand Index (KMeans vs Hierarchical):", result["ari"])
    print("\nDone. Figures saved to", FIG_DIR)
    print("Labelled data saved to", OUT_DIR / "clustered_clients.csv")


if __name__ == "__main__":
    main()
