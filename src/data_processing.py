"""
data_processing.py
-------------------
Loads, cleans and merges the Parcl clients & properties datasets, then
engineers a client-level feature table used for clustering.

Author: Buyer Segmentation & Investment Profiling project
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------
def load_raw(data_dir: Path = DATA_DIR):
    clients = pd.read_csv(data_dir / "clients.csv")
    properties = pd.read_csv(data_dir / "properties.csv")
    return clients, properties


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """The raw date_of_birth / transaction_date columns mix DD-MM-YYYY and
    M/D/YYYY style strings. Try a couple of explicit formats before
    falling back to pandas' inference so nothing silently becomes NaT."""
    parsed = pd.to_datetime(series, format="%m-%d-%Y", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(series[mask], format="%m/%d/%Y", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(series[mask], errors="coerce", dayfirst=False)
    return parsed


def clean_clients(clients: pd.DataFrame) -> pd.DataFrame:
    df = clients.copy()

    # Normalise column text / whitespace
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Drop exact duplicate rows and duplicate client_ids (keep first)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="client_id", keep="first")

    # Parse date_of_birth -> age (as of analysis date)
    df["date_of_birth"] = _parse_mixed_dates(df["date_of_birth"])
    analysis_date = pd.Timestamp("2025-12-31")
    df["age"] = ((analysis_date - df["date_of_birth"]).dt.days / 365.25).round(1)

    # Guard against unrealistic ages caused by parsing edge-cases
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median())

    # Normalise categorical labels (consistent casing)
    for c in ["client_type", "gender", "country", "region",
              "acquisition_purpose", "loan_applied", "referral_channel"]:
        df[c] = df[c].str.title()

    df["loan_applied_flag"] = (df["loan_applied"].str.lower() == "yes").astype(int)

    return df


def clean_properties(properties: pd.DataFrame) -> pd.DataFrame:
    df = properties.copy()

    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    df = df.drop_duplicates()

    # sale_price arrives as "$300,385.62" -> float
    df["sale_price"] = (
        df["sale_price"].replace(r"[\$,]", "", regex=True).astype(float)
    )

    df["transaction_date"] = _parse_mixed_dates(df["transaction_date"])
    df["unit_category"] = df["unit_category"].str.title()
    df["listing_status"] = df["listing_status"].str.title()

    return df


# --------------------------------------------------------------------------
# Feature engineering (client-level table used for clustering)
# --------------------------------------------------------------------------
def build_client_features(clients: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the transaction-level properties table up to one row per
    client and join it onto the client demographic/attitudinal fields."""

    sold = properties[properties["listing_status"] == "Sold"].copy()

    agg = sold.groupby("client_ref").agg(
        num_purchases=("listing_id", "count"),
        total_spend=("sale_price", "sum"),
        avg_purchase_price=("sale_price", "mean"),
        max_purchase_price=("sale_price", "max"),
        total_floor_area=("floor_area_sqft", "sum"),
        first_purchase_date=("transaction_date", "min"),
        last_purchase_date=("transaction_date", "max"),
        n_apartment=("unit_category", lambda s: (s == "Apartment").sum()),
        n_office=("unit_category", lambda s: (s == "Office").sum()),
        n_towers=("tower_number", "nunique"),
    ).reset_index().rename(columns={"client_ref": "client_id"})

    agg["is_multi_unit_buyer"] = (agg["num_purchases"] > 1).astype(int)
    agg["tenure_days"] = (agg["last_purchase_date"] - agg["first_purchase_date"]).dt.days
    agg["pct_office_units"] = (agg["n_office"] / agg["num_purchases"]).round(3)

    features = clients.merge(agg, on="client_id", how="left")

    # Clients present in clients.csv but with no completed ("Sold") transaction
    # (e.g. only reserved/available listings, or no linked listing at all)
    zero_cols = ["num_purchases", "total_spend", "avg_purchase_price",
                 "max_purchase_price", "total_floor_area", "n_apartment",
                 "n_office", "n_towers", "is_multi_unit_buyer", "tenure_days",
                 "pct_office_units"]
    for c in zero_cols:
        features[c] = features[c].fillna(0)

    features["has_purchase"] = (features["num_purchases"] > 0).astype(int)

    return features


def get_processed_data(data_dir: Path = DATA_DIR):
    clients_raw, properties_raw = load_raw(data_dir)
    clients = clean_clients(clients_raw)
    properties = clean_properties(properties_raw)
    features = build_client_features(clients, properties)
    return clients, properties, features


if __name__ == "__main__":
    clients, properties, features = get_processed_data()
    print("Clients:", clients.shape)
    print("Properties:", properties.shape)
    print("Client feature table:", features.shape)
    print(features.head())
