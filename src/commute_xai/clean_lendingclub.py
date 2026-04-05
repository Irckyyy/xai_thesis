import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# =========================================
# CLEANING FUNCTION
# =========================================
def clean_lending_club_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Original shape: {df.shape}")

    # -----------------------------
    # 1. Validate required column
    # -----------------------------
    if 'loan_status' not in df.columns:
        raise ValueError("Missing required column: loan_status")

    # -----------------------------
    # 2. Keep only final outcomes
    # -----------------------------
    final_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(final_statuses)].copy()
    print(f"After filtering final statuses: {df.shape}")

    # -----------------------------
    # 3. Create target label
    # -----------------------------
    df['derived_risk'] = df['loan_status'].apply(
        lambda x: 0 if x == 'Fully Paid' else 1
    )

    # Remove original target (avoid leakage)
    df.drop(columns=['loan_status'], inplace=True)

    # -----------------------------
    # 4. Convert percentage columns
    # -----------------------------
    percent_cols = ['int_rate', 'revol_util']
    for col in percent_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.rstrip('%')
                .replace('', np.nan)
                .astype(float)
            )

    # -----------------------------
    # 5. Handle dates → feature
    # -----------------------------
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')

    if 'issue_d' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')

    if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
        df['credit_history_months'] = (
            (df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12 +
            (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
        )
        df['credit_history_months'] = df['credit_history_months'].clip(lower=0)

    # Drop raw date columns
    df.drop(columns=[col for col in ['issue_d', 'earliest_cr_line'] if col in df.columns],
            inplace=True)

    # -----------------------------
    # 6. Keep relevant features (Set A)
    # -----------------------------
    keep_cols = [
        'loan_amnt', 'term','int_rate', 'installment', 'emp_length', 'home_ownership', 'annual_inc',
        'verification_status', 'purpose', 'dti', 'delinq_2yrs',
        'inq_last_6mths', 'open_acc', 'total_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'tot_cur_bal', 'tot_coll_amt',
        'application_type', 'credit_history_months',
        'derived_risk'
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].copy()
    print(f"After column selection: {df.shape}")

    # -----------------------------
    # 7. Handle missing values
    # -----------------------------
    # Categorical → "Unknown"
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    # Numeric → median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'derived_risk' in numeric_cols:
        numeric_cols.remove('derived_risk')

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # -----------------------------
    # 8. Basic sanity filters
    # -----------------------------
    if 'annual_inc' in df.columns:
        df = df[df['annual_inc'] > 0]

    if 'dti' in df.columns:
        df = df[df['dti'] >= 0]

    if 'credit_history_months' in df.columns:
        df = df[df['credit_history_months'] >= 0]

    print(f"Final cleaned shape: {df.shape}")

    return df


# =========================================
# STRATIFIED SAMPLING FUNCTION
# =========================================
def stratified_sample(df, target_size=20000, random_seed=42):
    print("Performing stratified sampling...")

    if len(df) <= target_size:
        print("Dataset already small. Skipping sampling.")
        return df.copy()

    df_sampled, _ = train_test_split(
        df,
        train_size=target_size,
        stratify=df['derived_risk'],
        random_state=random_seed
    )

    print(f"Sampled shape: {df_sampled.shape}")
    print("Class distribution:")
    print(df_sampled['derived_risk'].value_counts(normalize=True))

    return df_sampled


from pathlib import Path

# =========================================
# MAIN EXECUTION
# =========================================
if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    input_path = BASE_DIR / 'data' / 'loan.csv'
    processed_dir = BASE_DIR / 'data' / 'processed'

    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_output_path = processed_dir / 'lending_club_cleaned.csv'
    sampled_output_path = processed_dir / 'lending_club_cleaned_20k.csv'

    print(f"Reading data from: {input_path}")

    # Check cleaning
    cleaned_data = clean_lending_club_data(input_path)

    # Save cleaned full dataset
    cleaned_data.to_csv(cleaned_output_path, index=False)
    print(f"Saved cleaned dataset to: {cleaned_output_path}")

    # Stratified sampling to 20k rows
    sampled_data = stratified_sample(cleaned_data, target_size=20000)

    # Save sampled dataset
    sampled_data.to_csv(sampled_output_path, index=False)
    print(f"Saved sampled dataset to: {sampled_output_path}")

    print("Pipeline complete.")