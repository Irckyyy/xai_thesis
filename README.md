# Explainable Artificial Intelligence for Loan Decisions Using Constraint-Aware Counterfactual Explanations

## Introduction
This repository contains the data processing, modeling, and evaluation pipeline for my undergraduate Computer Science thesis on Explainable AI (XAI) in financial decision-making.

The study addresses the issue of **computational dissonance** in XAI—where model explanations are technically correct but practically unusable. To resolve this, the proposed framework integrates:

- **Descriptive explanations** via SHAP (feature attribution)
- **Prescriptive explanations** via DiCE (counterfactual generation)

These are unified under a **Constraint-Aware Actionability Taxonomy**, ensuring that generated recourse is:
- Mathematically valid  
- Logically consistent  
- Realistically actionable in real-world financial contexts  

---

## Dataset

The dataset used is the LendingClub credit risk dataset from Kaggle:

https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset

### Status (as of April 6, 2026)
- Raw dataset size: ~880,000 records  
- Working dataset: **20,000 records (stratified sample)**  

A reduced dataset is used to:
- Improve computational efficiency (especially for DiCE)
- Maintain class balance
- Ensure stable and reproducible experiments  

---

## Data Processing & Cleaning Methodology

To ensure valid model learning and reliable XAI outputs, strict preprocessing steps were applied.

### 1. Derived Attributes

Two key features were engineered:

- **`derived_risk`**  
  A binary target variable replacing `loan_status`:
  - `0` → Fully Paid  
  - `1` → Charged Off / Default  
  - All non-final statuses were removed  

- **`credit_history_months`**  
  Computed as the difference between `earliest_cr_line` and `issue_d`, representing the borrower’s credit age at application time.

---

### 2. Prevention of Data Leakage

All features containing **post-loan information** were removed to prevent the model from learning future outcomes.

Including such variables would:
- Artificially inflate model performance  
- Invalidate SHAP explanations  
- Produce unrealistic counterfactual recourse  

---

### 3. Dropped Columns

The following categories were removed:

#### a. Future Leakage Variables
- out_prncp, out_prncp_inv  
- total_pymnt, total_pymnt_inv  
- total_rec_prncp, total_rec_int  
- total_rec_late_fee  
- recoveries, collection_recovery_fee  
- last_pymnt_amnt  
- last_pymnt_d, next_pymnt_d  
- last_credit_pull_d  

#### b. Identifiers & Text Noise
- id, member_id, url, desc, title, policy_code  

#### c. Sparse / Joint Features (>90% missing)
- annual_inc_joint  
- dti_joint  
- verification_status_joint  

#### d. Original Targets / Raw Dates
- loan_status  
- issue_d  
- earliest_cr_line  

---

### 4. Handling Missing Values

To maintain dataset integrity:

- **Categorical features** → filled with `"Unknown"`  
- **Numerical features** → filled with median values  

---

### 5. Data Validation & Filtering

Invalid or nonsensical records were removed:

- `annual_inc <= 0`  
- `dti < 0`  
- `credit_history_months < 0`  

---

### 6. Stratified Sampling

A stratified sampling approach was applied to produce a **20,000-record dataset** while preserving the original class distribution of `derived_risk`.

---

## Setup Instructions

Due to GitHub file size limitations, the dataset is not included in this repository.

### Step 1: Download the Dataset
- Visit the Kaggle link above  
- Download and extract the dataset  

---

### Step 2: Add the Dataset

Place the file in the following directory:

xai-thesis/
│
├── data/
│ ├── loan.csv
│ └── processed/
│
├── src/
│ └── clean_lendingclub.py
├── .gitignore
└── README.md


---

### Step 3: Run the Pipeline

python src/clean_lendingclub.py

The pipeline generates:


- `lending_club_cleaned.csv` → Fully cleaned dataset  
- `lending_club_cleaned_20k.csv` → Stratified sample (used for experiments)  

---

## Notes

- This repository is intended for **academic thesis use**  
- All collaborators must use the **same dataset version**  
- The pipeline assumes consistent schema from the Kaggle dataset  

---