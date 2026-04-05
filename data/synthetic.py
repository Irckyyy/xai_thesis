import numpy as np
import pandas as pd

def generate_credit_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    # Age
    age = np.random.randint(21, 65, n_samples)

    # Gender
    gender = np.random.choice(["Male", "Female"], n_samples)

    # Employment length (years)
    employment_length = np.random.randint(0, 30, n_samples)

    # Credit history length (years)
    credit_history_length = np.maximum(
        np.random.normal(loc=age - 20, scale=5, size=n_samples).astype(int), 0
    )

    # Income (annual)
    income = np.round(np.random.normal(50000, 20000, n_samples), 2)
    income = np.clip(income, 10000, None)

    # Savings balance
    savings_balance = np.round(np.random.normal(20000, 15000, n_samples), 2)
    savings_balance = np.clip(savings_balance, 0, None)

    # Loan amount requested
    loan_amount_requested = np.round(np.random.normal(15000, 8000, n_samples), 2)
    loan_amount_requested = np.clip(loan_amount_requested, 1000, None)

    # Number of existing loans
    number_of_existing_loans = np.random.randint(0, 5, n_samples)

    # Default history
    default_history = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])

    # Debt-to-income ratio
    dti = (loan_amount_requested / income) + (number_of_existing_loans * 0.05)
    debt_to_income_ratio = np.round(dti, 2)

    # Loan approval probability model
    approval_score = (
        (income / 100000) +
        (savings_balance / 50000) +
        (credit_history_length / 30) +
        (employment_length / 20)
        - (debt_to_income_ratio)
        - (default_history * 1.5)
    )

    probability = 1 / (1 + np.exp(-approval_score))

    loan_approval = (probability > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "employment_length": employment_length,
        "credit_history_length": credit_history_length,
        "income": income,
        "debt_to_income_ratio": debt_to_income_ratio,
        "savings_balance": savings_balance,
        "loan_amount_requested": loan_amount_requested,
        "number_of_existing_loans": number_of_existing_loans,
        "default_history": default_history,
        "loan_approval": loan_approval
    })

    return df


# Generate dataset
dataset = generate_credit_dataset(1000)

# Save to CSV
dataset.to_csv("synthetic_credit_dataset.csv", index=False)

print(dataset.head())