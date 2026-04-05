import pandas as pd

# Load the dataset
df = pd.read_csv("data/loan.csv")

# Check all unique loan_status values first
print("Unique loan_status values:")
print(df["loan_status"].dropna().unique())
print()

# Keep only the statuses you want to use
selected_statuses = ["Fully Paid", "Charged Off", "Default"]
df_filtered = df[df["loan_status"].isin(selected_statuses)].copy()

# Map to binary classes
df_filtered["target"] = df_filtered["loan_status"].map({
    "Fully Paid": "Favorable",
    "Charged Off": "Unfavorable",
    "Default": "Unfavorable"
})

# Count each class
class_counts = df_filtered["target"].value_counts()
print("Class counts:")
print(class_counts)
print()

# Get percentages
class_percentages = df_filtered["target"].value_counts(normalize=True) * 100
print("Class percentages:")
print(class_percentages.round(2))
print()

# Optional: show original loan_status counts too
status_counts = df_filtered["loan_status"].value_counts()
status_percentages = df_filtered["loan_status"].value_counts(normalize=True) * 100

print("Filtered loan_status counts:")
print(status_counts)
print()

print("Filtered loan_status percentages:")
print(status_percentages.round(2))

import pandas as pd

# Load the dataset
df = pd.read_csv("data/loan.csv")

selected_statuses = ["Fully Paid", "Charged Off", "Default"]
df_filtered = df[df["loan_status"].isin(selected_statuses)].copy()

df_filtered["target"] = df_filtered["loan_status"].map({
    "Fully Paid": "Favorable",
    "Charged Off": "Unfavorable",
    "Default": "Unfavorable"
})

class_percentages = (df_filtered["target"].value_counts(normalize=True) * 100).round(2)

fav_pct = class_percentages.get("Favorable", 0)
unfav_pct = class_percentages.get("Unfavorable", 0)

print(f"After filtering the dataset, approximately {fav_pct}% of the records were classified as favorable and {unfav_pct}% as unfavorable.")