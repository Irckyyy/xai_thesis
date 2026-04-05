import pandas as pd
from pathlib import Path

# =========================================
# DATASET DIAGNOSTIC HEALTH CHECK SCRIPT
# =========================================
if __name__ == "__main__":

    # 1. Setup Paths dynamically
    BASE_DIR = Path(__file__).resolve().parent
    
    processed_dir = BASE_DIR.parent / 'data' / 'processed'
    
    full_cleaned_path = processed_dir / 'lending_club_cleaned.csv'
    sampled_path = processed_dir / 'lending_club_cleaned_20k.csv'

    print("Loading datasets for health check...")
    try:
        df_full = pd.read_csv(full_cleaned_path, low_memory=False)
        df_sample = pd.read_csv(sampled_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error loading files. Ensure paths are correct: {e}")
        exit()

    print("\n" + "="*50)
    print("1. TARGET DISTRIBUTION (CRITICAL)")
    print("="*50)
    print("Full Dataset Distribution:")
    print(df_full['derived_risk'].value_counts(normalize=True) * 100)
    print("\nSampled Dataset Distribution:")
    print(df_sample['derived_risk'].value_counts(normalize=True) * 100)

    print("\n" + "="*50)
    print("2. MISSING VALUES CHECK (VERY IMPORTANT)")
    print("="*50)
    missing_counts = df_sample.isnull().sum()
    if missing_counts.sum() == 0:
        print("ALL CLEAR: No missing values in the sampled dataset.")
    else:
        print("WARNING: Missing values detected!")
        print(missing_counts[missing_counts > 0])

    print("\n" + "="*50)
    print("3 & 5. FEATURE RANGES & ENGINEERED FEATURES")
    print("="*50)
    critical_cols = ['annual_inc', 'dti', 'revol_util', 'credit_history_months']
    
    for col in critical_cols:
        if col in df_sample.columns:
            print(f"\n--- {col} ---")
            print(f"Min: {df_sample[col].min()}, Max: {df_sample[col].max()}")
            print(f"Mean: {df_sample[col].mean():.2f}")
            
            # Check for negative values 
            if df_sample[col].min() < 0:
                print(f"RED FLAG: Negative values found in {col}!")
            if col == 'revol_util' and df_sample[col].max() > 100:
                print(f"NOTE: revol_util over 100% found. (Can happen legitimately, but monitor it)")
            if col == 'dti' and df_sample[col].max() > 100:
                print(f"NOTE: Extreme DTI over 100 found.")
        else:
            print(f"{col} is missing from the dataset!")

    print("\n" + "="*50)
    print("4. CATEGORICAL VALUES CHECK")
    print("="*50)
    categorical_cols = df_sample.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_vals = df_sample[col].unique()
        print(f"\n{col} (Unique count: {len(unique_vals)}):")
        # Print up to the first 10 unique values to check
        print(unique_vals[:10])

    print("\n" + "="*50)
    print("6. LEAKAGE COLUMNS CHECK")
    print("="*50)
    leakage_list = [
        'loan_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 
        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
        'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'
    ]
    
    found_leakage = set(leakage_list).intersection(set(df_sample.columns))
    if len(found_leakage) == 0:
        print("ALL CLEAR: No known leakage columns detected.")
    else:
        print(f"FATAL RED FLAG: Leakage columns found! -> {found_leakage}")

    print("\n" + "="*50)
    print("7 & 8. DATASET SIZE CONSISTENCY")
    print("="*50)
    print(f"Full Cleaned Shape: {df_full.shape}")
    print(f"Sampled Shape:      {df_sample.shape}")
    
    if df_sample.shape[0] == 20000:
        print("Sampled dataset has exactly 20,000 rows.")
    else:
        print(f"WARNING: Sampled dataset has {df_sample.shape[0]} rows, expected 20,000.")

    print("\n=== DIAGNOSTIC COMPLETE ===")