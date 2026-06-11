"""
SuperKart Sales Forecast API
Flask backend that serves the trained Gradient Boosting pipeline.

Endpoints:
  GET  /             → health check
  POST /v1/sales     → single product-store prediction
  POST /v1/salesbatch → batch prediction from CSV upload
"""

import joblib
import pandas as pd
from flask import Flask, request, jsonify

# ── Initialise Flask app ──────────────────────────────────────────────────────
app = Flask("SuperKart Sales Forecast API")

# ── Load the trained model pipeline ──────────────────────────────────────────
# The .joblib file bundles both the preprocessor and the model
model = joblib.load("superkart_sales_model_v1_0.joblib")

# Feature names expected by the model
EXPECTED_FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_Type", "Product_MRP", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Store_Age"
]


# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecast API! POST to /v1/sales for predictions."


# ── Single prediction endpoint ────────────────────────────────────────────────
@app.route("/v1/sales", methods=["POST"])
def predict_sales():
    """
    Expects a JSON body with all EXPECTED_FEATURES as keys.
    Note: send Store_Age (integer), not Store_Establishment_Year.
    Returns: {"Predicted_Sales": <float>}
    """
    data = request.get_json()

    # Build a single-row DataFrame in the correct column order
    sample = {feat: data[feat] for feat in EXPECTED_FEATURES}
    input_df = pd.DataFrame([sample])

    # Predict
    prediction = model.predict(input_df)[0]

    return jsonify({"Predicted_Sales": round(float(prediction), 2)})


# ── Batch prediction endpoint ─────────────────────────────────────────────────
@app.route("/v1/salesbatch", methods=["POST"])
def predict_sales_batch():
    """
    Expects a multipart/form-data upload with key 'file' containing a CSV.
    CSV must have the same EXPECTED_FEATURES columns.
    Returns: JSON list of records with an added 'Predicted_Sales' column.
    """
    file = request.files["file"]
    input_df = pd.read_csv(file)

    # Predict
    predictions = model.predict(input_df)
    input_df["Predicted_Sales"] = predictions.round(2)

    return jsonify(input_df.to_dict(orient="records"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
