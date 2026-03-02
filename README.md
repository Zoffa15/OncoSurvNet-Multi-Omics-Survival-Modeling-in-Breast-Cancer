# OncoSurvNet-Multi-Omics-Survival-Modeling-in-Breast-Cancer
Multi-omics survival modeling framework integrating RNA-seq, mutation, and clinical data from TCGA BRCA. Includes Cox models, DeepSurv neural networks, cross-validation, and an interactive Streamlit dashboard.


## Project Overview

OncoSurvNet is a translational AI framework for survival modeling in breast cancer using multi-omics data from TCGA (via cBioPortal).

The project integrates:

RNA-seq expression (Z-scored)

Somatic mutation profiles

Clinical survival outcomes

It implements:

* Principal Component Analysis (PCA)

* Univariate survival screening (volcano-style)

* Multivariate Cox proportional hazards modeling

* Cross-validation with concordance index

* DeepSurv neural survival modeling (PyTorch)

* Interactive Streamlit dashboard for exploration and risk prediction

## Quick Start

```bash
pip install -r requirements.txt
python source/preprocessing.py
python source/deepsurv_model.py
streamlit run dashboard/app.py
```

## Motivation

Traditional survival models often rely on a single molecular layer. However, cancer is driven by complex genomic and transcriptional interactions.

This project demonstrates how multi-omics integration can improve:

* Risk stratification

* Biomarker discovery

* Model generalization

* Translational interpretability

Designed with industry oncology applications in mind.

## Disclaimer
This project is for research and educational purposes only.
It is not intended for clinical decision-making.

## Dataset

Source:

* TCGA Breast Cancer (PanCancer Atlas)

Accessed via cBioPortal

Data modalities:

* RNA-seq (RSEM Z-scores relative to tumor samples)

Mutation MAF

* Clinical survival data (OS_MONTHS, OS_STATUS)

## Methods
1. Dimensionality Reduction

PCA to explore transcriptional variance structure

2. Biomarker Screening

Univariate Cox models

Hazard ratio + p-value visualization

3. Multivariate Modeling

Cox proportional hazards regression

Cross-validated C-index

4. Deep Learning Survival

DeepSurv neural Cox model

Early stopping

Time-dependent concordance index

5. Risk Stratification

Median split risk groups

Kaplan–Meier survival separation

## Performance Metrics

* Concordance Index (C-index)

* Time-dependent C-index (DeepSurv)

* Cross-validation

* Hazard Ratios

## Interactive Dashboard

The Streamlit application allows:

PCA visualization

Gene selection for modeling

Real-time hazard ratio calculation

Survival curve visualization

Patient-level risk scoring

To run locally:

>  pip install -r requirements.txt

>  streamlit run app.py

# Multi-Omics Survival Modeling in TCGA BRCA

An end-to-end multi-omics survival modeling framework integrating RNA expression, somatic mutation, and clinical survival data from TCGA Breast Cancer (BRCA).

This project demonstrates production-style machine learning development for oncology analytics, including preprocessing pipelines, feature selection, Cox modeling, DeepSurv neural survival networks, cross-validation, and an interactive Streamlit dashboard.

---

## Project Objective

To build and evaluate multi-omics survival prediction models that:

- Integrate transcriptomic and genomic features
- Account for censoring using survival-aware methods
- Evaluate performance using concordance index
- Enable interactive risk exploration through a dashboard

This framework mirrors real-world pharma analytics workflows for biomarker discovery and patient risk stratification.

---

## Data Source

Data derived from:

**TCGA Breast Invasive Carcinoma (BRCA) – PanCancer Atlas 2018**  
Accessed via cBioPortal.

Required files:
- Clinical survival data
- RNA-seq expression (Z-scores, tumor-referenced)
- Somatic mutation data

Raw TCGA data is not redistributed in this repository.  
Users must download directly from cBioPortal and place in `data/raw/`.

---

## Data Governance

Raw TCGA data is not redistributed in this repository.
Users must download data directly from cBioPortal.

All preprocessing steps are fully reproducible using provided scripts.

## Repository Structure


```
multiomics-survival-brca/
├── README.md
├── requirements.txt
├── config/config.yaml
├── data/
├── src/__init__.py
├── models/
│   ├── cox_model.pkl
│   ├── deepsurv_features.pt
│   ├── deepsurv_imputer.pt
│   ├── deepsurv_model.pt
│   ├── deepsurv_pipeline.pt
│   └── deepsurv_scaler.pkl
│   └── deepsurv_weights.pt
├── notebooks/code.ipynb
└── dashboard/app.py
```
---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/multiomics-survival-brca.git
cd multiomics-survival-brca
```

### Create virtual environment 

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Download from cBioPortal:

- `data_clinical_patient.txt`
- `data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt`
- `data_mutations.txt`

Place inside:

```
data/raw/
```

---

## Run Preprocessing

```bash
python src/preprocessing.py
```

Output:
```
data/processed/multiomics_merged.parquet
```

Preprocessing includes:

- Clinical survival cleaning
- Z-score RNA alignment
- Variance filtering
- Mutation binary encoding
- Multi-omics merge

---

## Train Cox Model

```bash
python src/cox_model.py
```

Outputs:
```
models/cox_model.pkl
```

Includes:

- Univariate Cox screening
- Multivariate Cox model
- Cross-validation (5-fold)
- Concordance index evaluation

---

## Train DeepSurv Neural Survival Model

```bash
python src/deepsurv_model.py
```

Outputs:
```
models/deepsurv_model.pt
models/scaler.pkl
```

DeepSurv architecture:
- Multi-layer perceptron
- Batch normalization
- Dropout regularization
- Early stopping
- Time-dependent C-index evaluation

---

## Launch Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard features:

- PCA explorer
- Biomarker hazard exploration
- Interactive Cox modeling
- Risk stratification (Kaplan–Meier)
- DeepSurv-based patient risk scoring

---

## Model Evaluationm!!!!!!!!

Primary metric:
- Time-dependent Concordance Index (C-index)

Example performance (illustrative):

| Model               | Mean CV C-index |
|---------------------|-----------------|
| DeepSurv Model                  | 0.67        |
| Multivariate Cox (Top 20 Genes) | 0.65        |
| Multi-Omics (Lasso Regularized | 0.64         |
| Multi-Omics (L2 Regularized) | 0.63           |
| Multi-Omics (Robust Genes)  | 0.62            |

DeepSurv demonstrates improved nonlinear modeling capacity while maintaining censor-awareness.

---

## Technical Highlights

- Z-score standardized RNA expression
- Variance-based gene filtering
- Multi-omics integration (mutation + expression)
- Censor-aware survival modeling
- Cross-validation for generalization assessment
- Modular, reproducible pipeline
- Config-driven hyperparameters
- Model artifact persistence
- Interactive clinical-style dashboard

---

## Methodological Notes

- Survival outcome: Overall Survival (OS)
- Censoring properly handled via Cox partial likelihood
- No accuracy metric used (not appropriate for survival tasks)
- Feature scaling applied before neural training
- No raw TCGA data redistributed

---


## License

This project is licensed under the MIT License.

You are free to:
- Use
- Modify
- Distribute
- Apply commercially

Attribution is required.


## Author

Zofia Olszewska


## Contact

For questions, collaboration, or discussion:
sofie.olszewska@gmail.com

---




