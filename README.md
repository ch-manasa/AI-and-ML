# AI-and-ML
This repository will contain the hands-on project works I have done for the coursework: Artificial Intelligence and Machine Learning. 

## Hackathons:
1) Risk Classification using LLMs
   This project focuses on building a risk classification system using prompt engineering with a pre-trained Large Language Model (LLM). The objective is to classify risk events i     into two categories: Cybersecurity (0) and Financial (1) based on textual data.

   Using a labeled dataset of historical risk events, the model is designed to:

   a) Accurately identify the type of risk from unstructured text
   b) Support proactive risk management and decision-making
   c) Improve organizational resilience and compliance

   The solution leverages LLM-based prompt engineering to perform classification on unseen data, with performance evaluated using accuracy.


------------------------------------------------------
## Capstone Projects:
Below is the objective Context on the projects.

## 1) FoodHub :
The food aggregator company has stored the data of the different orders made by the registered customers in their online portal. They want to analyze the data to get a fair idea about the demand of different restaurants which will help them in enhancing their customer experience. Suppose you are hired as a Data Scientist in this company and the Data Science team has shared some of the key questions that need to be answered. Perform the data analysis to find answers to these questions that will help the company improve its business.

## 2)  Personal Loan Campaign: 
To predict whether a liability customer will buy personal loans, to understand which customer attributes are most significant in driving purchases, and to identify which segment of customers to target more.

## 3) Easy Visa :
In FY 2016, the OFLC processed 775,979 employer applications for 1,699,957 positions for temporary and permanent labor certifications. This was a nine percent increase in the overall number of processed applications from the previous year. The process of reviewing every case is becoming a tedious task as the number of applicants is increasing every year.
    
The increasing number of applicants every year calls for a Machine Learning based solution that can help in shortlisting the candidates having higher chances of VISA approval. OFLC has hired the firm EasyVisa for data-driven solutions. You as a data scientist at EasyVisa have to analyze the data provided and, with the help of a classification model:
    
Facilitate the process of visa approvals.
Recommend a suitable profile for the applicants for whom the visa should be certified or denied based on the drivers that significantly influence the case status.

## 4) Wind Turbine Failure Prediction (Predictive Maintenance):
This project focuses on building machine learning models to predict wind turbine generator failures using sensor data. With 40 features derived from environmental and turbine    conditions, the goal is to enable predictive maintenance—identifying potential failures before they occur.

By accurately detecting failures:
   a) True Positives help schedule cost-effective repairs
   b) False Negatives (missed failures) are minimized to avoid expensive replacements
   c) False Positives are controlled to reduce unnecessary inspections
   
Multiple classification models are trained, tuned, and evaluated to find the most cost-efficient solution for real-world deployment.

## 5) SafeGaurd Corp - CCN:
A deep learning image classification project built for SafeGuard Corp to automatically detect whether workers are wearing safety helmets in workplace images.
Trained and compared four models on 4,125 images (200×200 RGB) to classify workers as **With Helmet** or **Without Helmet**, with a focus on minimising false negatives given the safety-critical nature of the task.
   a) Stratified 70/15/15 train-val-test split preserving the 3.3:1 class imbalance ratio
   b) Data augmentation (rotation, flip, zoom, shift) to simulate real CCTV conditions
   c) Evaluation prioritised **Recall for the Without Helmet class** over accuracy
   d) EarlyStopping used across all models to prevent overfitting

## 6) SuperKart — Sales Revenue Forecasting & Deployment:
An end-to-end machine learning project built for SuperKart, a retail chain 
operating across Tier 1, 2, and 3 cities, to forecast product-level sales 
revenue and deploy the solution for real-time use.
Using historical product and store data, the model is designed to:
   a) Accurately predict total sales revenue per product per store
   b) Support inventory management and regional sales strategy decisions
   c) Deliver forecasts via a live REST API and interactive web application

Multiple ensemble models (Random Forest, Gradient Boosting) were trained, 
tuned, and evaluated inside sklearn pipelines. The best model (Tuned Gradient 
Boosting, RMSE: 281.54, R²: 0.93) was serialized and deployed on Hugging Face 
Spaces using Flask (backend) and Streamlit (frontend), containerized via Docker.

- **Live App:** https://manasa92-superkart-frontend.hf.space
- **Backend API:** https://manasa92-superkart-backend.hf.space

## 7) [New Wheels SQL Analysis](./New_Wheels_SQL_Analysis) :
Business analysis of vehicle resale company using 
SQL — window functions, aggregations, QoQ revenue analysis, customer feedback trends using SQLite

## 8) Visit with Us – Wellness Tourism Package Prediction

An end-to-end MLOps project that predicts whether a customer is likely to purchase the Wellness Tourism Package before being contacted. The project includes automated data validation, preprocessing, feature engineering, model training with hyperparameter tuning using XGBoost, MLflow experiment tracking, GitHub Actions-based CI/CD automation, and deployment of the trained model as an interactive Streamlit web application for real-time predictions.


## 9) Flykite Airlines HR Policy Q&A Bot (LLM + RAG)

Capstone project for the AI & ML program. A prototype that answers employee HR policy questions from the Flykite Airlines HR handbook using an open-source LLM and Retrieval-Augmented Generation (RAG).

## Approach
1. **Baseline LLM:** Qwen2.5-7B-Instruct (4-bit) with no access to the handbook.
2. **Prompt engineering:** 5 system-prompt strategies, with two refined further.
3. **RAG:** handbook cleaned, chunked (1000/150), embedded with all-MiniLM-L6-v2, stored in ChromaDB, retrieved with similarity search (k=3).
4. **Hyperparameter tuning:** 7 configurations (chunk size, k, similarity vs MMR, temperature) plus a page-merged chunking experiment.

## Evaluation
- LLM judge for relevance and groundedness (1 to 5)
- Manual review against handbook reference answers
- Key-fact recall: share of expected policy facts present in each answer

## Results
| Method | Relevance | Groundedness |
|---|---|---|
| Base LLM | 4.2 | 3.2 |
| Prompt-engineered (P2 v3) | 4.2 | 3.4 |
| RAG (manual review) | 4.4 | 4.8 |
| RAG tuned (cfg1, judge) | 5.0 | 5.0 |

The selected configuration (1000/150, k=3, similarity) had the highest key-fact recall (0.91). The main remaining risk is page-based chunking splitting multi-part rules; section-based chunking is the recommended next step.

## Tech stack
Python, Hugging Face Transformers, bitsandbytes, LangChain, ChromaDB, sentence-transformers, Google Colab (T4 GPU)

## Running the notebook
Open in Google Colab with a T4 GPU runtime. Run the install cell, restart the session, then run all cells. The HR handbook PDF is not included in this repo; update `pdf_path` to point to your own copy.
