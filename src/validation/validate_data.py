import pandas as pd
import great_expectations as gx


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv("data/raw/telco.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# Great Expectations
# ============================================================

gx_df = gx.dataset.PandasDataset(df)


# ============================================================
# Expectations
# ============================================================

checks = []

# tenure should be between 0 and 100
checks.append(
    gx_df.expect_column_values_to_be_between(
        "tenure",
        min_value=0,
        max_value=100
    )
)

# MonthlyCharges should not be null
checks.append(
    gx_df.expect_column_values_to_not_be_null(
        "MonthlyCharges"
    )
)

# Churn should only contain Yes/No
checks.append(
    gx_df.expect_column_values_to_be_in_set(
        "Churn",
        ["Yes", "No"]
    )
)

# Churn should not be null
checks.append(
    gx_df.expect_column_values_to_not_be_null(
        "Churn"
    )
)

# customerID should not be null
checks.append(
    gx_df.expect_column_values_to_not_be_null(
        "customerID"
    )
)


# ============================================================
# Validation Results
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

all_checks_passed = True

for check in checks:

    success = check["success"]

    print(
        f"{check['expectation_config']['expectation_type']}: "
        f"{'PASSED' if success else 'FAILED'}"
    )

    if not success:
        all_checks_passed = False


# ============================================================
# Validation Gate
# ============================================================

print("\n" + "=" * 60)

if all_checks_passed:

    print("ALL DATA VALIDATION CHECKS PASSED")
    print("Pipeline can continue.")

else:

    print("DATA VALIDATION FAILED")
    print("Pipeline stopped.")

    raise ValueError(
        "Data validation failed. "
        "Training pipeline stopped."
    )