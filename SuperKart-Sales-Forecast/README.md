# SuperKart — Sales Revenue Forecasting & Deployment

A end-to-end machine learning project to forecast product-level sales revenue 
for SuperKart, a retail chain operating across Tier 1, 2, and 3 cities.

## Project Structure
- `notebook/` — EDA, preprocessing, model building, and evaluation
- `backend/` — Flask REST API serving the trained model
- `frontend/` — Streamlit web app for interactive predictions
- `data/` — Raw dataset
- `model/` — Serialized model pipeline (.joblib)

## Models Built
Random Forest and Gradient Boosting trained inside sklearn pipelines.
Tuned Gradient Boosting selected as final model — Test RMSE: 281.54, R²: 0.93.

## Live Demo
- Backend API: https://manasa92-superkart-backend.hf.space
- Frontend App: https://manasa92-superkart-frontend.hf.space

## Tech Stack
Python · scikit-learn · XGBoost · Flask · Streamlit · Docker · Hugging Face Spaces
