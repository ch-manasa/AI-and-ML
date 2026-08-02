
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

# Paths are relative to the repo root - this script runs both locally (with cwd
# set to the Drive project folder) and inside GitHub Actions (with cwd = repo root
# after actions/checkout), so no absolute Drive path is used here.
project_root_path = "tourism_project"
model_building_path = os.path.join(project_root_path, "model_building")
MODEL_OUT_DIR = os.path.join(project_root_path, "deployment")
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "best_model_v1.joblib")


def load_splits():
    Xtrain = pd.read_csv(os.path.join(model_building_path, "Xtrain.csv"))
    Xtest = pd.read_csv(os.path.join(model_building_path, "Xtest.csv"))
    ytrain = pd.read_csv(os.path.join(model_building_path, "ytrain.csv")).squeeze("columns")
    ytest = pd.read_csv(os.path.join(model_building_path, "ytest.csv")).squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numerical_features = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                random_state=1,
                eval_metric="logloss",
                verbosity=0,
            )),
        ]
    )
    return pipeline


def main():
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    Xtrain, Xtest, ytrain, ytest = load_splits()
    print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")

    # Handle class imbalance (roughly 80/20 split of ProdTaken)
    neg, pos = np.bincount(ytrain)
    scale_pos_weight = neg / pos

    pipeline = build_pipeline(Xtrain)

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__scale_pos_weight": [1, scale_pos_weight],
    }

    mlflow.set_experiment("tourism-wellness-package-prediction")

    with mlflow.start_run():
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=3,
            n_jobs=1,
        )
        grid.fit(Xtrain, ytrain)

        best_model = grid.best_estimator_
        print("Best params:", grid.best_params_)

        # Log all tuned parameters to MLflow
        mlflow.log_params(grid.best_params_)

        # Evaluate on the test set
        preds = best_model.predict(Xtest)
        probs = best_model.predict_proba(Xtest)[:, 1]

        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds),
            "recall": recall_score(ytest, preds),
            "f1": f1_score(ytest, preds),
            "roc_auc": roc_auc_score(ytest, probs),
        }
        mlflow.log_metrics(metrics)

        print("Test set performance:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification report:\n", classification_report(ytest, preds))

    joblib.dump(best_model, MODEL_OUT_PATH)
    print(f"Best model saved to {MODEL_OUT_PATH}")

    with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
        mlflow.log_artifact(MODEL_OUT_PATH, artifact_path="model")


if __name__ == "__main__":
    main()
