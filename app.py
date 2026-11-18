import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="AutoClean & Cluster", layout="wide")
st.title("AutoClean & Unsupervised Clustering")
st.caption("Upload a single CSV file to clean features, inspect PCA variance, and segment rows with K-Means.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    accept_multiple_files=False,
)

k_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)


def clean_and_preprocess(raw_df: pd.DataFrame):
    stats = {
        "initial_rows": raw_df.shape[0],
        "initial_cols": raw_df.shape[1],
        "missing_cells": int(raw_df.isna().sum().sum()),
    }

    # 1. Remove completely empty rows and exact duplicates
    df = raw_df.dropna(how="all").drop_duplicates().copy()
    stats["duplicates_removed"] = stats["initial_rows"] - df.shape[0]

    # 2. Drop columns with >50% missing values or <=1 unique value
    valid_cols = [
        col for col in df.columns
        if df[col].isna().mean() <= 0.5 and df[col].nunique(dropna=True) > 1
    ]
    df = df[valid_cols]

    # 3. Keep numeric columns and low-cardinality categorical columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [
        col for col in df.select_dtypes(exclude=["number"]).columns
        if df[col].nunique(dropna=True) <= min(30, int(0.5 * len(df)))
    ]

    kept_cols = num_cols + cat_cols
    stats["dropped_cols"] = [c for c in raw_df.columns if c not in kept_cols]
    df_cleaned = df[kept_cols].copy()

    if not kept_cols:
        raise ValueError("No usable numeric or low-cardinality categorical columns remained after cleaning.")

    # 4. Build leak-free preprocessing pipeline
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(df_cleaned)

    # Extract readable feature names after scaling & one-hot encoding
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split("__", 1)[-1] for name in raw_feature_names]

    return df_cleaned, X_processed, feature_names, num_cols, cat_cols, stats


if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if len(raw_df) < 5:
            st.error("Dataset must contain at least 5 rows.")
            st.stop()

        df_cleaned, X_processed, feature_names, num_cols, cat_cols, stats = clean_and_preprocess(raw_df)

        # Fit Unsupervised Model (K-Means)
        effective_k = min(k_clusters, len(df_cleaned) - 1)
        kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_processed)

        try:
            sil_score = silhouette_score(X_processed, labels)
            sil_display = f"{sil_score:.3f}"
        except ValueError:
            sil_display = "N/A"

        # Full PCA Analysis
        pca = PCA(random_state=42)
        coords = pca.fit_transform(X_processed)
        explained_var = pca.explained_variance_ratio_ * 100
        cum_var = np.cumsum(explained_var)

        # Build Top 5 Principal Components summary
        n_top_pcs = min(5, pca.n_components_)
        pca_rows = []
        for i in range(n_top_pcs):
            loadings = pca.components_[i]
            top_idx = np.argsort(np.abs(loadings))[::-1][:3]
            top_drivers = ", ".join(
                f"{feature_names[idx]} ({loadings[idx]:+.2f})" for idx in top_idx
            )
            pca_rows.append({
                "Principal Component": f"PC{i + 1}",
                "Primary Feature": feature_names[top_idx[0]],
                "Explained Variance (%)": round(float(explained_var[i]), 2),
                "Cumulative Variance (%)": round(float(cum_var[i]), 2),
                "Top Contributing Features (Loadings)": top_drivers,
            })
        top_pca_df = pd.DataFrame(pca_rows)

        # Rank Top 5 Original Features driving overall variance across top components
        weighted_importance = np.sum(
            np.abs(pca.components_[:n_top_pcs]) * pca.explained_variance_ratio_[:n_top_pcs, None],
            axis=0,
        )
        top_feat_indices = np.argsort(weighted_importance)[::-1][:min(5, len(feature_names))]
        top_features_df = pd.DataFrame({
            "Rank": range(1, len(top_feat_indices) + 1),
            "Feature Name": [feature_names[idx] for idx in top_feat_indices],
            "PC1 Loading": [round(float(pca.components_[0, idx]), 3) for idx in top_feat_indices],
            "PC2 Loading": [
                round(float(pca.components_[1, idx]), 3) if pca.n_components_ > 1 else 0.0
                for idx in top_feat_indices
            ],
            "Weighted Variance Score": [round(float(weighted_importance[idx]), 4) for idx in top_feat_indices],
        })

        # 2D Scatter DataFrame
        pc1_var = explained_var[0]
        pc2_var = explained_var[1] if pca.n_components_ > 1 else 0.0
        plot_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1] if pca.n_components_ > 1 else 0.0,
            "Cluster": [f"Cluster {c}" for c in labels],
        })

        result_df = df_cleaned.copy()
        result_df.insert(0, "Cluster", labels)

        # --- Screen Output ---
        st.subheader("1. Data Cleaning Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Kept", f"{len(df_cleaned)} / {stats['initial_rows']}")
        c2.metric("Duplicates Removed", stats["duplicates_removed"])
        c3.metric("Missing Cells Imputed", stats["missing_cells"])
        c4.metric("Features Used", f"{len(num_cols)} num, {len(cat_cols)} cat")

        if stats["dropped_cols"]:
            st.caption(f"Dropped sparse/constant/ID columns: `{', '.join(stats['dropped_cols'])}`")

        st.subheader("2. PCA Analysis & Top 5 Components / Features")
        p1, p2, p3 = st.columns(3)
        p1.metric("Total Components", pca.n_components_)
        p2.metric("2D Variance (PC1 + PC2)", f"{cum_var[min(1, len(cum_var) - 1)]:.1f}%")
        p3.metric(f"Top {n_top_pcs} PCs Cumulative Variance", f"{cum_var[n_top_pcs - 1]:.1f}%")

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("**Top 5 Principal Components**")
            st.dataframe(top_pca_df, use_container_width=True, hide_index=True)
        with col_right:
            st.markdown("**Top 5 Driving Feature Names**")
            st.dataframe(top_features_df, use_container_width=True, hide_index=True)

        st.subheader("3. Clustering Results")
        m1, m2 = st.columns(2)
        m1.metric("Clusters (k)", effective_k)
        m2.metric(
            "Silhouette Score",
            sil_display,
            help="Ranges from -1 to 1. Higher indicates denser, better-separated clusters.",
        )

        st.caption(f"2D PCA Projection — PC1 ({pc1_var:.1f}% variance) vs. PC2 ({pc2_var:.1f}% variance)")
        st.scatter_chart(plot_df, x="PC1", y="PC2", color="Cluster", height=420)

        st.subheader("4. Average Values of Numerical Columns per Cluster")
        if num_cols:
            cluster_summary = result_df.groupby("Cluster")[num_cols].mean().round(3)
            cluster_summary.insert(0, "Row Count", result_df.groupby("Cluster").size())
            cluster_summary.index = [f"Cluster {idx}" for idx in cluster_summary.index]
            cluster_summary.index.name = "Cluster"
            st.dataframe(cluster_summary.reset_index(), use_container_width=True, hide_index=True)
        else:
            st.info("No numerical columns were present in the uploaded dataset.")

        st.subheader("5. Cleaned & Labeled Dataset")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download Clustered CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="clustered_output.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Could not process file: {e}")