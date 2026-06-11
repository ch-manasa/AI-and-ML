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
 
