📘 Employee Attrition Prediction App (2025 Edition)

A modern HR analytics tool powered by Machine Learning + SHAP explainability.

🚀 Overview

This Streamlit application predicts employee attrition probability using a trained Random Forest model.
It also provides transparent explanations using SHAP values so HR teams can understand:

Why an employee may leave

Which factors contribute the most

What changes could reduce risk

Designed for real-world HR decision-making, the app blends predictive accuracy with intuitive visual explanations.

🧠 Key Features
🔮 Attrition Risk Prediction

Enter employee details → the app predicts the probability of attrition.

📊 Explainable AI with SHAP

Shows the top 2–3 factors driving each prediction, so HR can take action.

🎛️ Interactive What-If Analysis

Users can adjust features and instantly see how risk changes.

🎨 Modern 2025 UI

Clean, elegant Streamlit layout designed for business users.

🏗️ Tech Stack
Layer	Technology
Frontend	Streamlit
Model	RandomForestClassifier
Explainability	SHAP
Data Processing	Pandas + NumPy
Deployment	Streamlit Cloud or Vercel (via Streamlit Serverless)
📦 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/employee-attrition-app.git
cd employee-attrition-app

2️⃣ Install dependencies

Create a requirements.txt containing:

streamlit
pandas
numpy
scikit-learn
shap


Then install:

pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

🤖 Machine Learning Model

The Random Forest model uses engineered features such as:

Age

MonthlyIncome

DistanceFromHome

YearsInCurrentRole

OverTime flag

PercentSalaryHike

Engagement Index

Tenure Ratio

WorkLifeBalance

EnvironmentSatisfaction

And more…

The model was exported to best_rf.pkl and loaded in the Streamlit app for predictions.

🔍 Explainability with SHAP

For every prediction, SHAP provides:

Positive contributors (increase attrition risk)

Negative contributors (reduce attrition risk)

Example output:

Top 3 reasons driving this risk
• OverTime_flag: +0.084
• Engagement Index: +0.053
• EnvironmentSatisfaction: −0.014

This helps managers understand why the model produces a given score.