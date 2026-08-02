import os
import sys
import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")  # relative to repo root

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: dataset is missing expected columns: {missing_cols}")
        sys.exit(1)

    print("Dataset Registration Summary")
    print("=" * 40)
    print(f"File path        : {DATA_PATH}")
    print(f"Rows             : {df.shape[0]}")
    print(f"Columns          : {df.shape[1]}")
    print(f"Expected columns : all {len(EXPECTED_COLUMNS)} present")
    print(f"Missing values   :\n{df.isnull().sum().sum()} total")
    print(f"Target balance   :\n{df['ProdTaken'].value_counts(normalize=True).round(3).to_dict()}")
    print("=" * 40)
    print("Dataset registered and validated successfully.")


if __name__ == "__main__":
    main()
