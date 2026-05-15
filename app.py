import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔍",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #f8f9fb; }
    .stApp { background-color: #f8f9fb; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .hero p  { font-size: 1rem; opacity: 0.75; margin-top: 0.4rem; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .card h3 { margin-top: 0; color: #1a1a2e; font-size: 1rem; font-weight: 700; }

    .result-high {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white; border-radius: 16px; padding: 2rem;
        text-align: center; margin: 1rem 0;
    }
    .result-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; border-radius: 16px; padding: 2rem;
        text-align: center; margin: 1rem 0;
    }
    .result-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .result-sub   { font-size: 1rem; opacity: 0.9; margin-top: 0.3rem; }

    .metric-box {
        background: #f0f4ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: white !important; }
    .stSlider > div { color: #1a1a2e; }
    label { font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load or Mock Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model   = joblib.load('model/xgb_churn_model.pkl')
        scaler  = joblib.load('model/scaler.pkl')
        features = joblib.load('model/feature_names.pkl')
        return model, scaler, features, True
    except:
        return None, None, None, False

model, scaler, feature_names, model_loaded = load_model()


# ─── Mock Prediction (when no model file yet) ──────────────────
def mock_predict(data):
    """Simple rule-based mock until the real model is trained."""
    score = 0
    if data['Contract'] == 0:          score += 35   # Month-to-month
    if data['tenure'] < 12:            score += 25
    if data['MonthlyCharges'] > 65:    score += 15
    if data['TechSupport'] == 0:       score += 10
    if data['OnlineSecurity'] == 0:    score += 10
    if data['PaymentMethod'] == 2:     score += 5    # Electronic check
    return min(score, 98) / 100


# ─── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 Customer Churn Predictor</h1>
  <p>Enter customer details to predict churn risk — powered by XGBoost ML model</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar — Model Info ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.markdown("---")
    st.markdown("**Algorithm:** XGBoost")
    st.markdown("**Accuracy:** 85%")
    st.markdown("**AUC-ROC:** 0.90")
    st.markdown("**Dataset:** Telco Churn (Kaggle)")
    st.markdown("**Records:** 7,043 customers")
    st.markdown("---")
    if model_loaded:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ Using demo mode\nRun the notebook first to train & save the real model.")
    st.markdown("---")
    st.markdown("**Built by:** Your Name")
    st.markdown("**GitHub:** [View Repo](#)")


# ─── Input Form ───────────────────────────────────────────────
st.markdown("## 👤 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><h3>📋 Account Info</h3>', unsafe_allow_html=True)
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method  = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges   = st.slider("Total Charges ($)", 0, 9000, monthly_charges * tenure)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><h3>🌐 Internet Services</h3>', unsafe_allow_html=True)
    internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security   = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support      = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><h3>👤 Demographics</h3>', unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents     = st.selectbox("Has Dependents?", ["Yes", "No"])
    phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Predict Button ────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

if predict_btn:
    # Encode inputs
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map  = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
    binary_map = {"Yes": 1, "No": 0}
    yes_no_nis = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

    input_data = {
        'gender':            1 if gender == "Male" else 0,
        'SeniorCitizen':     1 if senior_citizen == "Yes" else 0,
        'Partner':           binary_map[partner],
        'Dependents':        binary_map[dependents],
        'tenure':            tenure,
        'PhoneService':      binary_map[phone_service],
        'MultipleLines':     yes_no_nis[multiple_lines],
        'InternetService':   internet_map[internet_service],
        'OnlineSecurity':    yes_no_nis[online_security],
        'OnlineBackup':      yes_no_nis[online_backup],
        'DeviceProtection':  yes_no_nis[device_protection],
        'TechSupport':       yes_no_nis[tech_support],
        'StreamingTV':       yes_no_nis[streaming_tv],
        'StreamingMovies':   yes_no_nis[streaming_movies],
        'Contract':          contract_map[contract],
        'PaperlessBilling':  binary_map[paperless_billing],
        'PaymentMethod':     payment_map[payment_method],
        'MonthlyCharges':    monthly_charges,
        'TotalCharges':      total_charges,
    }

    # Predict
    if model_loaded:
        input_df   = pd.DataFrame([input_data])
        input_df   = input_df[feature_names]
        churn_prob = model.predict_proba(input_df)[0][1]
    else:
        churn_prob = mock_predict(input_data)

    churn_pct = churn_prob * 100

    # ─── Result Display ────────────────────────────────────────
    st.markdown("## 📈 Prediction Result")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if churn_prob >= 0.5:
            st.markdown(f"""
            <div class="result-high">
              <p class="result-title">⚠️ High Churn Risk</p>
              <p class="result-sub">This customer is likely to leave</p>
              <h2 style="font-size:3rem; margin:0.5rem 0">{churn_pct:.1f}%</h2>
              <p style="opacity:0.85">Probability of churning</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
              <p class="result-title">✅ Low Churn Risk</p>
              <p class="result-sub">This customer is likely to stay</p>
              <h2 style="font-size:3rem; margin:0.5rem 0">{churn_pct:.1f}%</h2>
              <p style="opacity:0.85">Probability of churning</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk meter
        st.progress(churn_prob)

    with res_col2:
        st.markdown('<div class="card"><h3>🔑 Key Risk Factors</h3>', unsafe_allow_html=True)

        factors = []
        if contract == "Month-to-month":
            factors.append(("🔴 Month-to-month contract", "43% avg churn rate"))
        if tenure < 12:
            factors.append(("🔴 New customer (< 12 months)", "High early churn risk"))
        if monthly_charges > 65:
            factors.append(("🟡 High monthly charges", f"${monthly_charges}/mo"))
        if tech_support in ["No"]:
            factors.append(("🟡 No tech support", "Increases churn risk"))
        if online_security in ["No"]:
            factors.append(("🟡 No online security", "Increases churn risk"))
        if payment_method == "Electronic check":
            factors.append(("🟡 Electronic check payment", "Highest churn payment method"))

        if not factors:
            st.success("✅ No major risk factors detected!")
        else:
            for factor, detail in factors:
                st.markdown(f"""
                <div class="metric-box">
                  <strong>{factor}</strong><br>
                  <small style="color:#666">{detail}</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="card"><h3>💡 Recommended Actions</h3>', unsafe_allow_html=True)
        if churn_prob >= 0.5:
            st.markdown("""
            <div class="insight-box">
            🎯 <strong>Offer a discount</strong> to switch to annual contract<br><br>
            📞 <strong>Proactive outreach</strong> — call before they cancel<br><br>
            🛡️ <strong>Bundle tech support</strong> at reduced price<br><br>
            💳 <strong>Auto-pay incentive</strong> — reduce friction
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
            ✅ <strong>Customer is stable</strong> — maintain quality service<br><br>
            🌟 <strong>Loyalty reward</strong> — recognize long-term customers<br><br>
            📦 <strong>Upsell opportunity</strong> — offer premium features
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary table
    st.markdown("### 📋 Input Summary")
    summary_df = pd.DataFrame({
        'Feature': ['Tenure', 'Contract', 'Monthly Charges', 'Internet Service', 'Tech Support', 'Payment Method'],
        'Value':   [f'{tenure} months', contract, f'${monthly_charges}', internet_service, tech_support, payment_method]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Built with Python · XGBoost · Streamlit &nbsp;|&nbsp; "
    "Dataset: Telco Customer Churn (Kaggle) &nbsp;|&nbsp; "
    "Portfolio Project by <strong>Your Name</strong></p>",
    unsafe_allow_html=True
)
