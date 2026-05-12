import pandas as pd
import great_expectations as gx


# Load dataset
df = pd.read_csv("data/raw/telco.csv")


# Create GX dataframe
gx_df = gx.dataset.PandasDataset(df)


# -----------------------------
# EXPECTATIONS
# -----------------------------

# tenure should be between 0 and 100
expect_tenure = gx_df.expect_column_values_to_be_between(
    "tenure",
    min_value=0,
    max_value=100
)

# MonthlyCharges should not be null
expect_monthly = gx_df.expect_column_values_to_not_be_null(
    "MonthlyCharges"
)

# Churn values should only contain Yes/No
expect_churn = gx_df.expect_column_values_to_be_in_set(
    "Churn",
    ["Yes", "No"]
)

# -----------------------------
# VALIDATION GATE
# -----------------------------

all_checks_passed = (
    expect_tenure["success"]
    and expect_monthly["success"]
    and expect_churn["success"]
)

if all_checks_passed:
    print("\n All data validation checks passed!")

else:
    raise ValueError(
        " Data validation failed! Pipeline stopped."
    )
