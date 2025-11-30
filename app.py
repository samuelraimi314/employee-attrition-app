import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

# -------------------------------
# Load model and SHAP explainer
# -------------------------------
with open("best_rf.pkl", "rb") as f:
    best_rf = pickle.load(f)

with open("explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for Modern Design
# -------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Card Containers */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 48px;
        font-weight: 700;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 40px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Risk Indicator */
    .risk-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(245, 87, 108, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .risk-value {
        font-size: 72px;
        font-weight: 800;
        color: white;
        margin: 20px 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    
    .risk-label {
        font-size: 24px;
        color: white;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.08);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.12);
        border-left-width: 6px;
        transform: translateX(5px);
    }
    
    .feature-name {
        color: #667eea;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .feature-value {
        color: white;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Text Styling */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    p, label, span, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Input Fields */
    .stNumberInput input, .stSelectbox select {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 10px !important;
    }
    
    /* Slider */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Header Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 56px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 20px;
        font-weight: 400;
        margin-bottom: 40px;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .status-high {
        background: rgba(245, 87, 108, 0.2);
        border: 1px solid #f5576c;
        color: #f5576c;
    }
    
    .status-medium {
        background: rgba(250, 177, 92, 0.2);
        border: 1px solid #fab15c;
        color: #fab15c;
    }
    
    .status-low {
        background: rgba(52, 211, 153, 0.2);
        border: 1px solid #34d399;
        color: #34d399;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar - Employee Inputs
# -------------------------------
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>⚙️ Employee Profile</h2>", unsafe_allow_html=True)

def user_input_features():
    st.sidebar.markdown("### 👤 Personal Information")
    age = st.sidebar.number_input("Age", 18, 60, 35)
    distance = st.sidebar.number_input("Distance From Home (km)", 0, 50, 10)
    
    st.sidebar.markdown("### 💰 Compensation")
    income = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 6000)
    percent_salary_hike = st.sidebar.number_input("Percent Salary Hike", 0, 100, 15)
    
    st.sidebar.markdown("### 📅 Career History")
    total_working_years = st.sidebar.number_input("Total Working Years", 0, 40, 10)
    years_at_company = st.sidebar.number_input("Years at Company", 0, 40, 5)
    years_in_current_role = st.sidebar.number_input("Years in Current Role", 0, 20, 3)
    tenure_ratio = st.sidebar.number_input("Tenure Ratio", 0.0, 1.0, 0.5)
    
    st.sidebar.markdown("### 😊 Satisfaction Metrics")
    job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
    env_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
    work_life_balance = st.sidebar.slider("Work Life Balance", 1, 4, 3)
    engagement_index = st.sidebar.slider("Engagement Index", 0.0, 1.0, 0.7)
    
    st.sidebar.markdown("### ⏰ Work Conditions")
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    overtime_low_hike = st.sidebar.selectbox("Overtime & Low Hike", [0, 1])
    
    data = {
        "Age": age,
        "DistanceFromHome": distance,
        "MonthlyIncome": income,
        "TotalWorkingYears": total_working_years,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": env_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "OverTime_flag": 1 if overtime=="Yes" else 0,
        "tenure_ratio": tenure_ratio,
        "engagement_index": engagement_index,
        "PercentSalaryHike": percent_salary_hike,
        "overtime_low_hike": overtime_low_hike
    }
    return pd.DataFrame(data, index=[0])

employee_df = user_input_features()

# -------------------------------
# Main Page - Header
# -------------------------------
st.markdown("<div class='gradient-text'>🎯 Employee Attrition Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Predictive Analytics for Workforce Retention</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction & SHAP
# -------------------------------
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔮 Attrition Risk Assessment")
    
    if st.button("🚀 Analyze Employee Risk"):
        with st.spinner("Analyzing employee data..."):
            risk = best_rf.predict_proba(employee_df)[:,1][0]
            
            # Determine risk level
            if risk >= 0.7:
                status_class = "status-high"
                status_text = "HIGH RISK"
                emoji = "🚨"
            elif risk >= 0.4:
                status_class = "status-medium"
                status_text = "MODERATE RISK"
                emoji = "⚠️"
            else:
                status_class = "status-low"
                status_text = "LOW RISK"
                emoji = "✅"
            
            st.markdown(f"""
            <div class='risk-container'>
                <div class='risk-label'>{emoji} Attrition Probability</div>
                <div class='risk-value'>{risk*100:.1f}%</div>
                <span class='status-badge {status_class}'>{status_text}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # SHAP explanation
            st.markdown("### 🎯 Key Contributing Factors")
            st.markdown("<p style='opacity: 0.8; margin-bottom: 20px;'>Top 3 features driving this prediction:</p>", unsafe_allow_html=True)
            
            shap_values_employee = explainer(employee_df)
            shap_vals_pos = shap_values_employee.values[0][:,1]
            top_idx = np.argsort(np.abs(shap_vals_pos))[::-1][:3]
            
            for i, idx in enumerate(top_idx, 1):
                impact = "Increases" if shap_vals_pos[idx] > 0 else "Decreases"
                color = "#f5576c" if shap_vals_pos[idx] > 0 else "#34d399"
                st.markdown(f"""
                <div class='feature-card'>
                    <div class='feature-name'>#{i} {employee_df.columns[idx]}</div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div class='feature-value' style='color: {color};'>{shap_vals_pos[idx]:.3f}</div>
                        <span style='color: {color}; font-weight: 600;'>{impact} Risk</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Employee Overview")
    
    # Create metric cards
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Tenure</div>
        <div class='metric-value'>{employee_df['YearsAtCompany'].values[0]}</div>
        <div class='metric-label'>Years</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
        <div class='metric-label'>Monthly Income</div>
        <div class='metric-value'>${employee_df['MonthlyIncome'].values[0]:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
        <div class='metric-label'>Engagement Score</div>
        <div class='metric-value'>{employee_df['engagement_index'].values[0]:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.markdown("### 💡 Quick Insights")
    st.markdown("""
    <p style='font-size: 14px; line-height: 1.8;'>
    • Adjust employee parameters in the sidebar<br>
    • Click 'Analyze' to generate predictions<br>
    • Review key factors driving attrition risk<br>
    • Take proactive retention actions
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 30px; background: rgba(255, 255, 255, 0.03); border-radius: 15px; border-top: 1px solid rgba(255, 255, 255, 0.1);'>
    <p style='font-size: 14px; opacity: 0.7; margin: 0;'>Powered by Advanced Machine Learning • Built with ❤️ for Data-Driven HR</p>
    <p style='font-size: 12px; opacity: 0.5; margin-top: 10px;'>© 2025 Employee Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)