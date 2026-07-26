# 🏢 Machine Learning Based Buyer Segmentation & Investment Profiling for Real Estate Market Intelligence

An end-to-end unsupervised machine learning project that segments real estate buyers into
actionable personas and profiles their investment behaviour, built for **Parcl Co.
Limited** as part of the **Unified Mentor** data science program.

The project cleans and merges raw client + transaction data, engineers a client-level
behavioural feature set, clusters buyers with **K-Means** (validated against
**Hierarchical/Agglomerative clustering**), and ships the results as a live, filterable
**Streamlit dashboard** plus a written research paper.

---

## ✨ What this project does

- **Cleans & merges** two raw CSVs (2,000 clients × 10,000 property transactions) —
  handling mixed date formats, currency-formatted prices, duplicates, and missing values.
- **Engineers** a client-level feature table (age, financing behaviour, spend, purchase
  count, price tier, etc.).
- **Clusters buyers** into 4 data-driven segments — *Global Investors*, *Personal-Use
  Buyers*, *First-Time Buyers*, *Luxury Investors* — using K-Means, with the optimal `k`
  chosen via the Elbow Method + Silhouette Score, and validated against an independent
  Hierarchical Clustering run (Adjusted Rand Index reported).
- **Visualizes** everything: EDA charts, elbow/silhouette curves, PCA cluster scatter, a
  dendrogram, and segment-level behavioural breakdowns.
- **Ships an interactive Streamlit dashboard** with 4 modules (Segmentation Overview,
  Investor Behaviour, Geographic Analysis, Segment Insights) and filters by country,
  region, acquisition purpose, and client type.
- **Documents everything** in a full research paper (`reports/research_paper.md`) with
  EDA, methodology, findings, and business recommendations.

---

## 📂 Project Structure

```
buyer-segmentation/
├── app/
│   └── streamlit_app.py          # Interactive dashboard (4 tabs, sidebar filters)
├── data/
│   ├── clients.csv               # Raw client data (provided)
│   ├── properties.csv            # Raw property/transaction data (provided)
│   ├── clustered_clients.csv     # Generated: client table + cluster labels
│   ├── cluster_summary.csv       # Generated: per-segment summary statistics
│   └── k_evaluation.csv          # Generated: elbow/silhouette scores per k
├── models/
│   ├── preprocessor.joblib       # Generated: fitted ColumnTransformer (scaler + OHE)
│   ├── kmeans_model.joblib       # Generated: fitted K-Means model
│   └── pca.joblib                # Generated: fitted PCA (for 2D visualisation)
├── reports/
│   ├── research_paper.md         # Full write-up: EDA, methodology, findings, recommendations
│   └── figures/                  # 16 generated PNG charts referenced by the paper
├── src/
│   ├── data_processing.py        # Cleaning + feature engineering
│   ├── clustering.py             # Encoding, scaling, K-Means, Hierarchical, naming logic
│   └── run_analysis.py           # One-command pipeline: runs everything, saves outputs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/buyer-segmentation.git
cd buyer-segmentation
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 2. Run the analysis pipeline

This cleans the data, engineers features, fits the clustering models, and writes all
figures + labelled datasets used by the dashboard and research paper:

```bash
python src/run_analysis.py
```

Outputs land in `data/` (labelled dataset + summary tables), `models/` (fitted model
artefacts) and `reports/figures/` (16 PNG charts).

### 3. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

> **Note:** the dashboard reads `data/clustered_clients.csv`, so run step 2 at least once
> before launching it.

---

## 🧠 Methodology Summary

| Step | Approach |
|---|---|
| Data cleaning | Duplicate removal, mixed-format date parsing, currency parsing, missing-value handling |
| Feature engineering | Age, financing flag, purchase count/spend/price aggregates per client |
| Encoding | One-Hot Encoding (`client_type`, `acquisition_purpose`, `referral_channel`, `country`) |
| Scaling | `StandardScaler` on numeric features |
| Clustering | K-Means (primary) + Agglomerative/Hierarchical (validation) |
| Choosing k | Elbow Method (inertia) + Silhouette Score, k=4 selected |
| Validation | Adjusted Rand Index between K-Means and Hierarchical labels (0.586) |

Full details, all 16 figures, and business recommendations are in
[`reports/research_paper.md`](reports/research_paper.md).

---

## 📊 The Four Buyer Segments

| Segment | Share | Avg. Total Spend | Loan-Applied Rate | Headline Trait |
|---|---|---|---|---|
| 🌍 Global Investors | 33.9% | $1.53M | 24.9% | High-value, largely cash-capable repeat buyers |
| 🏠 Personal-Use Buyers | 36.2% | $1.04M | 0.0% | Pay in full, lowest average price point |
| 🔑 First-Time Buyers | 27.6% | $1.11M | 100.0% | Youngest segment, fully financing-dependent |
| 💎 Luxury Investors | 2.4% | $2.47M | 31.9% | Highest spend & satisfaction, premium relationship priority |

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy for data wrangling
- **scikit-learn** — StandardScaler, OneHotEncoder, KMeans, AgglomerativeClustering, PCA, silhouette_score
- **scipy** — hierarchical dendrogram
- **matplotlib / seaborn** — static EDA & clustering figures
- **Streamlit + Plotly** — interactive dashboard
- **joblib** — model persistence

---

## 📄 License

This project was built for educational/portfolio purposes as part of the Unified Mentor
program in collaboration with Parcl Co. Limited. Add your preferred license (e.g. MIT)
here before publishing if required.

---

## 🙋 Acknowledgements

Project brief & dataset provided by **Unified Mentor** and **Parcl Co. Limited**.
