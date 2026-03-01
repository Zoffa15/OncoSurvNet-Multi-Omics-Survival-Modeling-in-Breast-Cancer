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

## Motivation

Traditional survival models often rely on a single molecular layer. However, cancer is driven by complex genomic and transcriptional interactions.

This project demonstrates how multi-omics integration can improve:

* Risk stratification

* Biomarker discovery

* Model generalization

* Translational interpretability

Designed with industry oncology applications in mind.

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

## Future Improvements

* External validation on METABRIC

* SHAP interpretability for DeepSurv

* Pathway-level modeling

* Multi-modal neural fusion

* Batch effect correction

* Deployment via Docker

## Author

Zofia Olszewska

Data Scientist 
