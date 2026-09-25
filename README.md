# Agentic AI Framework for Autonomous Data Analysis and Insight Generation

## Architecture
Dataset + user objective
-> Profile Agent
-> Planner Agent
-> EDA Agent
-> Model Agent
-> Validation Agent
-> Insight Agent

## Autonomous task selection
The Planner Agent now determines whether the uploaded tabular dataset should be analyzed as:

- **Classification**: categorical/discrete target
- **Regression**: continuous numeric target
- **Clustering**: no target or an explicit grouping/segmentation objective

The UI also provides an optional target-column override when automatic target detection is ambiguous.

## Model selection
For supervised tasks, the Model Agent compares simple baselines and selects the best candidate using the appropriate metric:

- Classification: Logistic Regression vs Random Forest Classifier; Accuracy and weighted F1
- Regression: Linear Regression vs Random Forest Regressor; MAE, RMSE and R2
- Clustering: K-Means with candidate values of k; Silhouette Score

The system no longer treats every numeric target as classification. For example, a continuous `Age` target is treated as a regression problem.

## Run
pip install -r requirements.txt
copy .env.example to .env
streamlit run app.py

## Important limitation
Automatic target detection is heuristic when the user does not explicitly provide a target. For a production-quality system, the planner should surface its confidence and ask the user to confirm an ambiguous target rather than silently making a questionable choice.

## Why this is agentic
The Planner Agent uses the dataset profile and user objective to select the analytical task and executable workflow. Specialized agents then perform profiling, EDA, modeling/clustering, validation, and insight synthesis.
