"""
clustering.py
-------------
Encodes, scales and clusters the client feature table produced by
data_processing.py. Provides:
  - Elbow / Silhouette evaluation to pick k
  - K-Means clustering (primary segmentation)
  - Agglomerative (hierarchical) clustering (validation)
  - Human-readable cluster interpretation / naming
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "age", "satisfaction_score", "loan_applied_flag",
    "num_purchases", "total_spend", "avg_purchase_price",
    "total_floor_area", "pct_office_units", "is_multi_unit_buyer",
]

CATEGORICAL_FEATURES = [
    "client_type", "acquisition_purpose", "referral_channel", "country",
]
# `region` (57 distinct values) is intentionally excluded from the clustering
# feature set to avoid a very high-cardinality one-hot block dominating the
# distance metric relative to the dataset size. It is still used for the
# geographic dashboard and for post-hoc profiling of the resulting clusters.


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )


def prepare_matrix(features: pd.DataFrame, preprocessor: ColumnTransformer = None, fit=True):
    """Only clients with at least one completed purchase are meaningful to
    segment on investment behaviour, so we cluster that subset."""
    df = features[features["has_purchase"] == 1].copy().reset_index(drop=True)

    if preprocessor is None:
        preprocessor = build_preprocessor()
    if fit:
        X = preprocessor.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    else:
        X = preprocessor.transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    return df, X, preprocessor


def evaluate_k_range(X, k_range=range(2, 9), random_state=42):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return pd.DataFrame({"k": list(k_range), "inertia": inertias, "silhouette": silhouettes})


def fit_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def fit_hierarchical(X, k):
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(X)
    return agg, labels


def name_segments(df: pd.DataFrame, label_col="cluster") -> dict:
    """Data-driven business naming based on cluster-level averages.

    Exploration of this dataset shows `loan_applied` and total spend /
    average purchase price are the strongest separators between clusters
    (company_rate and acquisition_purpose turn out to be fairly evenly
    spread across clusters rather than cluster-defining) so naming leans
    on those signals, while still mapping onto the four buyer archetypes
    described in the project brief (Global Investors, First-Time Buyers,
    Corporate Buyers, Luxury Investors)."""
    summary = df.groupby(label_col).agg(
        avg_age=("age", "mean"),
        avg_spend=("total_spend", "mean"),
        avg_price=("avg_purchase_price", "mean"),
        loan_rate=("loan_applied_flag", "mean"),
        investment_rate=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        company_rate=("client_type", lambda s: (s == "Company").mean()),
        avg_satisfaction=("satisfaction_score", "mean"),
        multi_unit_rate=("is_multi_unit_buyer", "mean"),
        n_clients=("client_id", "count"),
    )

    size_25 = summary["n_clients"].quantile(0.25)
    # company_rate varies only marginally (~4-6%) across clusters in this
    # dataset -- client_type does not turn out to be cluster-defining here,
    # so a "Corporate Buyers" label is only used if a cluster's company
    # share is at least double the overall baseline; otherwise the more
    # informative behavioural signals (financing dependence + spend tier)
    # drive the name, ranking clusters by average spend.
    baseline_company_rate = df["client_type"].eq("Company").mean()
    spend_rank = summary["avg_spend"].rank(ascending=False, method="first")

    names = {}
    for cl, row in summary.iterrows():
        if row["company_rate"] >= 2 * baseline_company_rate:
            name = "Corporate Buyers"
        elif row["n_clients"] <= size_25 and spend_rank[cl] == 1:
            name = "Luxury Investors"
        elif row["loan_rate"] >= 0.5:
            name = "First-Time Buyers"
        elif spend_rank[cl] <= 2:
            name = "Global Investors"
        else:
            name = "Personal-Use Buyers"
        names[cl] = name

    # ensure uniqueness by appending cluster id if names collide
    seen = {}
    final = {}
    for cl, name in names.items():
        if name in seen:
            seen[name] += 1
            final[cl] = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
            final[cl] = name
    return final, summary


def run_full_pipeline(features: pd.DataFrame, k: int = 4, random_state: int = 42):
    df, X, preprocessor = prepare_matrix(features)

    eval_df = evaluate_k_range(X)

    kmeans_model, km_labels = fit_kmeans(X, k, random_state)
    hier_model, hier_labels = fit_hierarchical(X, k)

    ari = adjusted_rand_score(km_labels, hier_labels)

    df["cluster"] = km_labels
    df["cluster_hierarchical"] = hier_labels

    segment_names, summary = name_segments(df, "cluster")
    df["segment_name"] = df["cluster"].map(segment_names)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    df["pca_1"] = coords[:, 0]
    df["pca_2"] = coords[:, 1]

    # persist artifacts
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(pca, MODELS_DIR / "pca.joblib")

    return {
        "df": df,
        "eval_df": eval_df,
        "kmeans_model": kmeans_model,
        "hier_model": hier_model,
        "ari": ari,
        "segment_names": segment_names,
        "summary": summary,
        "preprocessor": preprocessor,
        "pca": pca,
    }


if __name__ == "__main__":
    from data_processing import get_processed_data

    _, _, features = get_processed_data()
    result = run_full_pipeline(features, k=4)
    print(result["eval_df"])
    print(result["segment_names"])
    print(result["summary"])
    print("Adjusted Rand Index (KMeans vs Hierarchical):", result["ari"])
