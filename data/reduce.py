import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data/loan.csv")

print(f"Original dataset shape: {df.shape}")

# Stratified sampling based on 'loan_status'
df_sampled, _ = train_test_split(
    df,
    train_size=100000,      # number of rows to sample
    stratify=df['loan_status'],  # maintain proportion of each loan_status category
    random_state=42
)

print(f"Stratified sampled dataset shape: {df_sampled.shape}")

# Save to a new CSV
df_sampled.to_csv("100k_loan_sampled_stratified.csv", index=False)
print("Reduced stratified dataset saved as 100k_loan_sampled_stratified.csv")