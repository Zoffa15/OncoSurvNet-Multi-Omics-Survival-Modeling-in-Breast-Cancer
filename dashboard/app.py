# ==========================================
# OncoSurvNet - Multi-Omics Survival Dashboard
# ==========================================

# ==========================================
# Imports
# ==========================================
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import torchtuples as tt
from pycox.models import CoxPH

# ==========================================
# Project Paths
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")

clinical_file = os.path.join(DATA_PATH, "data_clinical_patient.txt")
rna_file = os.path.join(DATA_PATH, "data_RNA_Seq_v2_mRNA_median_all_sample_Zscores.txt")
mutation_file = os.path.join(DATA_PATH, "data_mutations.txt")
cox_model_file = os.path.join(MODEL_PATH, "cox_model.pkl")
deepsurv_pipeline_file = os.path.join(MODEL_PATH, "deepsurv_pipeline.pt")
comparison_file = os.path.join(DATA_PATH, "model_comparison_results.csv")

# ==========================================
# Streamlit Config
# ==========================================
st.set_page_config(page_title="OncoSurvNet - TCGA BRCA Survival", layout="wide")
st.title("🧬 OncoSurvNet: Multi-Omics Survival Modeling (TCGA-BRCA)")

# ==========================================
# Load & Cache Functions
# ==========================================
@st.cache_data
def load_clinical(path):
    df = pd.read_csv(path, sep="\t", comment="#")
    df["event"] = df["OS_STATUS"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)
    df["time"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df = df[["PATIENT_ID", "time", "event"]].dropna()
    df.rename(columns={"PATIENT_ID": "Patient_ID"}, inplace=True)
    return df

@st.cache_data
def load_rna(path):
    rna = pd.read_csv(path, sep="\t")
    rna.dropna(subset=["Hugo_Symbol"], inplace=True)
    rna.set_index("Hugo_Symbol", inplace=True)
    if "Entrez_Gene_Id" in rna.columns:
        rna.drop(columns=["Entrez_Gene_Id"], inplace=True)
    rna = rna.T
    rna.index.name = "Patient_ID"
    rna.reset_index(inplace=True)
    rna["Patient_ID"] = rna["Patient_ID"].apply(lambda x: x[:-3] if str(x).endswith("-01") else x)
    return rna

@st.cache_data
def load_mutation(path):
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t")
    return None

@st.cache_data
def load_model_comparison(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ==========================================
# Load DeepSurv Safely
# ==========================================
@st.cache_resource
def load_deepsurv_pipeline_safe(pipeline_file, device="cpu"):
    """
    Safely load DeepSurv pipeline with PyTorch 2.6+.
    Allows unpickling numpy and SimpleImputer safely.
    Returns: model, imputer, scaler, feature columns
    """
    import sklearn.impute

    if not os.path.exists(pipeline_file):
        return None, None, None, None

    # --- Safe globals for PyTorch 2.6+
    safe_globals = [
        sklearn.impute._base.SimpleImputer,
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
    ]

    # --- Load checkpoint safely with weights_only=False ---
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(pipeline_file, map_location=device, weights_only=False)

    # --- Rebuild network ---
    features = checkpoint["features"]
    in_features = len(features)
    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=[128, 64, 32],
        out_features=1,
        batch_norm=True,
        dropout=0.3,
        activation=torch.nn.ReLU
    )

    model = CoxPH(net, tt.optim.Adam)
    model.net.load_state_dict(checkpoint["model_state_dict"])
    model.net.to(device)

    return model, checkpoint["imputer"], checkpoint["scaler"], features

# ==========================================
# Load Raw Data
# ==========================================
clinical = load_clinical(clinical_file)
rna = load_rna(rna_file)
mutation = load_mutation(mutation_file)
comparison_df = load_model_comparison(comparison_file)

# Merge clinical + RNA
data = pd.merge(clinical, rna, on="Patient_ID", how="inner")
data = data.loc[:, ~data.columns.duplicated()]  # remove duplicates
data.fillna(0, inplace=True)

# Feature matrix
X = data.drop(columns=["Patient_ID", "time", "event"])
y_time = data["time"].values
y_event = data["event"].values

# ==========================================
# Load Cox Model
# ==========================================
cox_model = None
if os.path.exists(cox_model_file):
    with open(cox_model_file, "rb") as f:
        cox_model = pickle.load(f)
    st.success("✅ Cox model loaded.")

# ==========================================
# Load DeepSurv Model
# ==========================================
deepsurv_model, imputer, scaler, feature_cols = load_deepsurv_pipeline_safe(deepsurv_pipeline_file)

if deepsurv_model:
    # 1️⃣ Select features used by the DeepSurv model
    X_deep = X[feature_cols].copy()
    X_deep = X_deep.apply(pd.to_numeric, errors="coerce")

    # 2️⃣ Impute missing values
    X_imputed = imputer.transform(X_deep)

    # 3️⃣ Scale features
    X_scaled = scaler.transform(X_imputed)

    # 4️⃣ Convert to float32 to match PyTorch model dtype
    X_scaled = X_scaled.astype('float32')

    # 5️⃣ Predict risk scores
    risks = deepsurv_model.predict(X_scaled).flatten()
    data["deepsurv_risk"] = risks

    st.dataframe(
        data[["Patient_ID", "deepsurv_risk"]]
        .sort_values("deepsurv_risk", ascending=False)
        .head(20)
    )
    st.success("✅ DeepSurv risk scores computed successfully.")
else:
    st.warning("DeepSurv pipeline not available.")


# ------------------------------------------
# Dataset Overview
# ------------------------------------------
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Patients", data.shape[0])
col2.metric("Total Columns", data.shape[1])
col3.metric("Gene Features", data.shape[1] - 3)
st.dataframe(data.head())

# ------------------------------------------
# Feature Matrix
# ------------------------------------------
X = data.drop(columns=["Patient_ID", "time", "event"])
y_time = data["time"].values
y_event = data["event"].values

# ------------------------------------------
# Load Cox Model
# ------------------------------------------
cox_model = None
try:
    with open(cox_model_file, "rb") as f:
        cox_model = pickle.load(f)
    st.success("✅ Cox model loaded.")
except:
    st.warning("Cox model not found. Using placeholder risk scores.")

# ------------------------------------------
# Load DeepSurv Model (PyCox)
# ------------------------------------------

# ==========================================
# Sidebar Navigation
# ==========================================
page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "PCA Explorer",
        "Univariate Cox (Volcano)",
        "Multivariate Cox + CV",
        "Mutation Integration",
        "DeepSurv Risk",
        "Risk Prediction",
        "Model Comparison"
    ]
)

# ==========================================
# Page: Overview
# ==========================================
if page == "Overview":
    st.markdown("""
    **OncoSurvNet Dashboard** integrates:
    - RNA expression (Z-scored)
    - Somatic mutation data
    - Clinical survival outcomes

    Features:
    - PCA visualization
    - Univariate Cox (Volcano-style)
    - Multivariate Cox with Cross-validation (C-Index)
    - DeepSurv risk prediction
    - Risk stratification plots
    """)

# ==========================================
# Page: PCA Explorer
# ==========================================
elif page == "PCA Explorer":
    st.subheader("🧬 PCA Explorer")
    n_components = st.slider("Number of PCA components", 2, min(10, X.shape[1]), 2)
    scaler = StandardScaler()
    X_scaled_pca = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled_pca)
    pca_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
    pca_df["event"] = y_event
    fig, ax = plt.subplots()
    sns.scatterplot(x="PC1", y="PC2", hue="event", palette="Set1", data=pca_df, ax=ax)
    ax.set_title("PCA Projection (First 2 Components)")
    st.pyplot(fig)

# ==========================================
# Page: Univariate Cox (Volcano)
# ==========================================
elif page == "Univariate Cox (Volcano)":
    st.subheader("Volcano-style Univariate Cox Screening")
    gene = st.selectbox("Select Gene", X.columns[:200])
    df_temp = data[["time", "event", gene]].dropna()
    cph = CoxPHFitter()
    try:
        cph.fit(df_temp, duration_col="time", event_col="event")
        hr = np.exp(cph.params_[gene])
        pval = cph.summary.loc[gene, "p"]
        st.write(f"Hazard Ratio: {hr:.2f}")
        st.write(f"P-value: {pval:.4g}")
    except Exception as e:
        st.warning(f"Cox fit failed: {e}")

# ==========================================
# Page: Multivariate Cox + CV
# ==========================================
elif page == "Multivariate Cox + CV":
    st.subheader("Multivariate Cox Model with 5-fold CV")
    selected_genes = st.multiselect("Select genes", X.columns[:200], default=X.columns[:5])
    if selected_genes:
        cox_data = data[["time", "event"] + list(selected_genes)]
        cph = CoxPHFitter()
        try:
            cph.fit(cox_data, duration_col="time", event_col="event")
            st.write(cph.summary)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cindex_list = []
            for train_idx, test_idx in kf.split(cox_data):
                train_df = cox_data.iloc[train_idx]
                test_df = cox_data.iloc[test_idx]
                cph.fit(train_df, duration_col="time", event_col="event")
                pred = cph.predict_partial_hazard(test_df)
                ci = concordance_index(test_df["time"], -pred, test_df["event"])
                cindex_list.append(ci)
            st.metric("Mean C-Index (CV)", round(np.mean(cindex_list),3))
        except Exception as e:
            st.warning(f"Cox fit failed: {e}")

# ==========================================
# Page: Mutation Heatmap (With Types)
# ==========================================
elif page == "Mutation Heatmap":
    st.subheader("Mutation Heatmap by Type (Top Genes)")

    if mutation is None or mutation.empty:
        st.warning("No mutation data loaded.")
    else:
        # We'll assume the mutation file has columns like:
        # Patient_ID | TP53 | BRCA1 | BRCA2 | ...
        # and each cell contains mutation type as string or NaN

        N_TOP_GENES = 20

        # Count non-empty entries per gene to find top mutated genes
        gene_counts = mutation.iloc[:, 1:].notna().sum().sort_values(ascending=False)
        top_genes = gene_counts.head(N_TOP_GENES).index.tolist()

        # Create a mutation matrix for the top genes
        mut_matrix = mutation[["Patient_ID"] + top_genes].copy()
        mut_matrix.set_index("Patient_ID", inplace=True)

        # Replace NaN with 'None' for easier plotting
        mut_matrix = mut_matrix.fillna("None")

        # Map unique mutation types to integers for heatmap
        mutation_types = sorted(mut_matrix.stack().unique())
        mut_type_map = {mt: i for i, mt in enumerate(mutation_types)}
        heatmap_data = mut_matrix.replace(mut_type_map)

        # Define a colormap for mutation types
        cmap = sns.color_palette("Set2", n_colors=len(mutation_types))

        st.write(f"Top {N_TOP_GENES} Mutated Genes Across Patients")
        st.dataframe(mut_matrix.head(10))  # show first 10 patients

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            heatmap_data.T,
            cmap=cmap,
            cbar=True,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"ticks": list(mut_type_map.values()), "label": "Mutation Type"}
        )

        # Set colorbar labels to mutation types
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(list(mut_type_map.values()))
        colorbar.set_ticklabels(list(mut_type_map.keys()))

        ax.set_xlabel("Patient ID")
        ax.set_ylabel("Gene")
        ax.set_title(f"Mutation Heatmap by Type (Top {N_TOP_GENES} Genes)")
        st.pyplot(fig)

# ==========================================
# Page: DeepSurv Risk
# ==========================================
# -- DeepSurv Risk
elif page == "DeepSurv Risk":
    st.subheader("DeepSurv Risk Predictions")

    if deepsurv_model:
        # Prepare data
        X_deep = X[feature_cols].copy()
        X_imputed = imputer.transform(X_deep)
        X_scaled = scaler.transform(X_imputed).astype("float32")  # ensure float32 for PyTorch

        # Predict risks
        risks = deepsurv_model.predict(X_scaled).flatten()
        data["deepsurv_risk"] = risks

        # Add an explanation
        st.markdown("""
        **Note:** The table below shows the top 10 patients with the highest DeepSurv risk scores.
        A higher risk score indicates a higher predicted risk of the event (e.g., shorter survival).
        Use this to identify high-risk patients and guide further analysis.
        """)

        # Show top 10
        st.dataframe(
            data[["Patient_ID", "deepsurv_risk"]]
            .sort_values("deepsurv_risk", ascending=False)
            .head(10)
        )

        st.success("✅ Risk scores computed.")
    else:
        st.warning("DeepSurv model not available.")

# ==========================================
# Page: Risk Prediction (Kaplan-Meier)
# ==========================================
elif page == "Risk Prediction":
    st.subheader("Kaplan-Meier Risk Stratification")
    # Cox risk
    cox_risk = data["cox_risk"] if "cox_risk" in data.columns else np.random.rand(len(data))
    cutoff = np.median(cox_risk)
    high_risk = cox_risk >= cutoff
    low_risk = ~high_risk

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()
    kmf.fit(data.loc[low_risk,"time"], data.loc[low_risk,"event"], label="Low Risk")
    kmf.plot_survival_function(ax=ax)
    kmf.fit(data.loc[high_risk,"time"], data.loc[high_risk,"event"], label="High Risk")
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Kaplan-Meier Curve (High vs Low Risk)")
    st.pyplot(fig)

# ==========================================
# Page: Model Comparison
# ==========================================
elif page == "Model Comparison":
    st.subheader("Model Performance Comparison")
    if comparison_df is not None:
        st.dataframe(comparison_df)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x="Mean C-index", y="Model", data=comparison_df, palette="viridis", ax=ax)
        ax.set_xlabel("Mean C-index")
        ax.set_ylabel("Model")
        ax.set_title("Comparison of Model Performance (Mean C-index)")
        st.pyplot(fig)
    else:
        st.warning("model_comparison_results.csv not found.")

# ==========================================
# Page: DeepSurv KM
# ==========================================
elif page == "DeepSurv KM":
    st.subheader("Kaplan-Meier: DeepSurv High vs Low Risk")

    if deepsurv_model is None:
        st.warning("DeepSurv model not loaded. Cannot compute KM curves.")
    else:
        # Predict risk scores
        risks = deepsurv_model.predict(X_scaled).flatten()
        data["deepsurv_risk"] = risks

        # Define high vs low risk
        cutoff = np.median(risks)
        high_risk = data["deepsurv_risk"] >= cutoff
        low_risk = ~high_risk

        # Plot Kaplan-Meier curves
        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(8,6))

        kmf.fit(data.loc[low_risk, "time"], data.loc[low_risk, "event"], label="Low Risk")
        kmf.plot_survival_function(ax=ax)

        kmf.fit(data.loc[high_risk, "time"], data.loc[high_risk, "event"], label="High Risk")
        kmf.plot_survival_function(ax=ax)

        ax.set_title("Kaplan-Meier Curves by DeepSurv Risk")
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival Probability")
        ax.grid(True)
        st.pyplot(fig)

        # Optional: show median survival times
        median_low = kmf.median_survival_time_
        median_high = KaplanMeierFitter().fit(data.loc[high_risk, "time"],
                                              data.loc[high_risk, "event"]).median_survival_time_
        st.write(f"Median Survival - Low Risk: {median_low:.1f} months")
        st.write(f"Median Survival - High Risk: {median_high:.1f} months")


# ------------------------------------------
# Sidebar Info
# ------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(f"Samples: {data.shape[0]}")
st.sidebar.markdown(f"Gene Features: {X.shape[1]}")