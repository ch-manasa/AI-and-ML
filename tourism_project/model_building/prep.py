import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths are relative to the repo root - this script runs both locally (with cwd
# set to the Drive project folder) and inside GitHub Actions (with cwd = repo root
# after actions/checkout), so no absolute Drive path is used here.
project_root_path = "tourism_project"
data_path = os.path.join(project_root_path, "data")
DATA_PATH = os.path.join(data_path, "tourism.csv")
model_building_path = os.path.join(project_root_path, "model_building")

TARGET_COL = "ProdTaken"

# Columns that carry no predictive signal (pure identifiers / row index)
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnecessary identifier columns if present
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Fix inconsistent category labels found during EDA
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows, if any
    df.drop_duplicates(inplace=True)

    return df


def main():
    os.makedirs(model_building_path, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded raw dataset: {df.shape}")

    df = clean_data(df)
    print(f"After cleaning: {df.shape}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    Xtrain.to_csv(os.path.join(model_building_path, "Xtrain.csv"), index=False)
    Xtest.to_csv(os.path.join(model_building_path, "Xtest.csv"), index=False)
    ytrain.to_csv(os.path.join(model_building_path, "ytrain.csv"), index=False)
    ytest.to_csv(os.path.join(model_building_path, "ytest.csv"), index=False)

    print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
    print(f"ytrain: {ytrain.shape}, ytest: {ytest.shape}")
    print("Train/test splits saved under tourism_project/model_building/")


if __name__ == "__main__":
    main()
