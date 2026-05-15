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

# ─── Custom CSS for visible hints ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #f8f9fb; }
    .stApp { background-color: #f8f9fb; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .hero p  { font-size: 1rem; opacity: 0.8; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .card h3 { margin-top: 0; color: #1a1a2e; font-size: 1rem; font-weight: 700; }

    /* Visible hint style */
    .hint {
        font-size: 0.75rem;
        color: #3b82f6;
        background: #eff6ff;
        padding: 4px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-top: 4px;
    }

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
    .result-sub   { font-size: 1rem; opacity: 0.9; }

    .metric-box {
        background: #f0f4ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        color: #1a1a2e;
    }
    .insight-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #1a1a2e;
    }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: white !important; }
    label { font-weight: 600 !important; color: #1e293b !important; }

    /* Force radio button labels to be visible */
    .stRadio > div {
        color: #1e293b !important;
    }
    .stRadio label {
        color: #1e293b !important;
    }
    div[role="radiogroup"] label {
        color: #1e293b !important;
    }
    .stRadio label span {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load or Mock Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/xgb_churn_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        features = joblib.load('model/feature_names.pkl')
        return model, scaler, features, True
    except:
        return None, None, None, False

model, scaler, feature_names, model_loaded = load_model()

def mock_predict(data):
    score = 0
    if data['Contract'] == 0: score += 35
    if data['tenure'] < 12: score += 25
    if data['MonthlyCharges'] > 65: score += 15
    if data['TechSupport'] == 0: score += 10
    if data['OnlineSecurity'] == 0: score += 10
    if data['PaymentMethod'] == 2: score += 5
    return min(score, 98) / 100

# ─── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 Customer Churn Predictor</h1>
  <p>Enter customer details to predict churn risk — powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.markdown("---")
    st.markdown("**Algorithm:** XGBoost | **Accuracy:** 85% | **AUC:** 0.90")
    st.markdown("**Dataset:** Telco Churn (7,043 customers)")
    st.markdown("---")
    if model_loaded:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ Demo mode – train the model first.")
    st.markdown("---")
    st.markdown("Built by [Awaliahftr](https://github.com/awaliahftr/churn-prediction)")

# ─── Main Input Form (Streamlined) ────────────────────────────
st.markdown("## 👤 Customer Profile")

# Use tabs to organize large number of inputs
tab1, tab2, tab3 = st.tabs(["📋 Account & Payment", "🌐 Internet Services", "👤 Demographics & Phone"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
        st.markdown("<span class='hint'>🔹 How many months with company</span>", unsafe_allow_html=True)
        
        contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
        st.markdown("<span class='hint'>🔹 Month-to-month has highest churn risk</span>", unsafe_allow_html=True)
        
        monthly_charges = st.slider("💰 Monthly Charges ($)", 18.0, 120.0, 65.0)
        st.markdown("<span class='hint'>🔹 Amount billed each month</span>", unsafe_allow_html=True)
        
    with col2:
        payment_method = st.selectbox("💳 Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)", 
            "Electronic check", "Mailed check"
        ])
        st.markdown("<span class='hint'>🔹 Electronic check = highest churn</span>", unsafe_allow_html=True)
        
        paperless_billing = st.radio("📧 Paperless Billing?", ["Yes", "No"], horizontal=True)
        st.markdown("<span class='hint'>🔹 Yes = bills by email only</span>", unsafe_allow_html=True)
        
        total_charges = monthly_charges * tenure
        st.metric("💵 Total Charges (auto-calculated)", f"${total_charges:,.0f}")

with tab2:
    st.caption("If customer has **No internet**, choose 'No' – other fields will be ignored.")
    
    internet_service = st.selectbox("🌐 Internet Service Type", ["DSL", "Fiber optic", "No"])
    st.markdown("<span class='hint'>🔹 Fiber optic is faster but often more expensive</span>", unsafe_allow_html=True)
    
    # Two columns for internet add-ons
    col_a, col_b = st.columns(2)
    with col_a:
        online_security = st.selectbox("🛡️ Online Security", ["Yes", "No", "No internet service"])
        st.markdown("<span class='hint'>🔹 Yes = antivirus/firewall add‑on</span>", unsafe_allow_html=True)
        
        online_backup = st.selectbox("💾 Online Backup", ["Yes", "No", "No internet service"])
        st.markdown("<span class='hint'>🔹 Yes = automatic cloud backup</span>", unsafe_allow_html=True)
        
        device_protection = st.selectbox("📱 Device Protection", ["Yes", "No", "No internet service"])
        st.markdown("<span class='hint'>🔹 Yes = insurance for equipment</span>", unsafe_allow_html=True)
        
    with col_b:
        tech_support = st.selectbox("🔧 Tech Support", ["Yes", "No", "No internet service"])
        st.markdown("<span class='hint'>🔹 Yes = 24/7 priority support</span>", unsafe_allow_html=True)
        
        streaming_tv = st.selectbox("📺 Streaming TV", ["Yes", "No", "No internet service"])
        st.markdown("<span class='hint'>🔹 Yes = live TV streaming</span>", unsafe_allow_html=True)
        
        streaming_movies = st.selectbox("🎬 Streaming Movies", ["Yes", "No", "No internet service"])
        st.markdown("<span class='hint'>🔹 Yes = on‑demand movie service</span>", unsafe_allow_html=True)

with tab3:
    col_x, col_y = st.columns(2)
    with col_x:
        gender = st.radio("👤 Gender", ["Male", "Female"], horizontal=True)
        st.markdown("<span class='hint'>🔹 Male/Female</span>", unsafe_allow_html=True)
        
        senior_citizen = st.radio("👵 Senior Citizen (65+)", ["No", "Yes"], horizontal=True)
        st.markdown("<span class='hint'>🔹 Yes = age 65 or older</span>", unsafe_allow_html=True)
        
        partner = st.radio("💑 Has Partner?", ["Yes", "No"], horizontal=True)
        st.markdown("<span class='hint'>🔹 Yes = lives with spouse/partner</span>", unsafe_allow_html=True)
        
        dependents = st.radio("👶 Has Dependents?", ["Yes", "No"], horizontal=True)
        st.markdown("<span class='hint'>🔹 Yes = children or dependents at home</span>", unsafe_allow_html=True)
        
    with col_y:
        phone_service = st.radio("📞 Phone Service?", ["Yes", "No"], horizontal=True)
        st.markdown("<span class='hint'>🔹 Yes = landline phone</span>", unsafe_allow_html=True)
        
        multiple_lines = st.selectbox("🔁 Multiple Lines", ["Yes", "No", "No phone service"])
        st.markdown("<span class='hint'>🔹 Yes = more than one phone line</span>", unsafe_allow_html=True)

# ─── Predict Button ────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

if predict_btn:
    # Encode inputs (same as before)
    contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
    payment_map = {"Bank transfer (automatic)":0, "Credit card (automatic)":1, "Electronic check":2, "Mailed check":3}
    binary_map = {"Yes":1, "No":0}
    yes_no_nis = {"Yes":1, "No":0, "No internet service":0, "No phone service":0}
    internet_map = {"DSL":0, "Fiber optic":1, "No":2}
    
    input_data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': binary_map[partner],
        'Dependents': binary_map[dependents],
        'tenure': tenure,
        'PhoneService': binary_map[phone_service],
        'MultipleLines': yes_no_nis[multiple_lines],
        'InternetService': internet_map[internet_service],
        'OnlineSecurity': yes_no_nis[online_security],
        'OnlineBackup': yes_no_nis[online_backup],
        'DeviceProtection': yes_no_nis[device_protection],
        'TechSupport': yes_no_nis[tech_support],
        'StreamingTV': yes_no_nis[streaming_tv],
        'StreamingMovies': yes_no_nis[streaming_movies],
        'Contract': contract_map[contract],
        'PaperlessBilling': binary_map[paperless_billing],
        'PaymentMethod': payment_map[payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }
    
    if model_loaded:
        input_df = pd.DataFrame([input_data])[feature_names]
        churn_prob = model.predict_proba(input_df)[0][1]
    else:
        churn_prob = mock_predict(input_data)
    
    churn_pct = churn_prob * 100
    
    # Display result
    st.markdown("## 📈 Prediction Result")
    res_col1, res_col2 = st.columns([1,1])
    
    with res_col1:
        if churn_prob >= 0.5:
            st.markdown(f"""
            <div class="result-high">
              <p class="result-title">⚠️ High Churn Risk</p>
              <p class="result-sub">This customer is likely to leave</p>
              <h2 style="font-size:3rem;">{churn_pct:.1f}%</h2>
              <p>Probability of churning</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
              <p class="result-title">✅ Low Churn Risk</p>
              <p class="result-sub">This customer is likely to stay</p>
              <h2 style="font-size:3rem;">{churn_pct:.1f}%</h2>
              <p>Probability of churning</p>
            </div>
            """, unsafe_allow_html=True)
        st.progress(float(churn_prob))
    
    with res_col2:
        st.markdown('<div class="card"><h3>🔑 Key Risk Factors</h3>', unsafe_allow_html=True)
        factors = []
        if contract == "Month-to-month": factors.append(("🔴 Month-to-month contract", "43% avg churn rate"))
        if tenure < 12: factors.append(("🔴 New customer (<12 months)", "High early churn risk"))
        if monthly_charges > 65: factors.append(("🟡 High monthly charges", f"${monthly_charges}/mo"))
        if tech_support == "No": factors.append(("🟡 No tech support", "Increases churn risk"))
        if online_security == "No": factors.append(("🟡 No online security", "Increases churn risk"))
        if payment_method == "Electronic check": factors.append(("🟡 Electronic check payment", "Highest churn method"))
        
        if not factors:
            st.success("✅ No major risk factors")
        else:
            for f, d in factors:
                st.markdown(f'<div class="metric-box"><strong>{f}</strong><br><small>{d}</small></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card"><h3>💡 Recommended Actions</h3>', unsafe_allow_html=True)
        if churn_prob >= 0.5:
            st.markdown("""
            <div class="insight-box">
            🎯 Offer discount to switch to annual contract<br>
            📞 Proactive outreach before cancellation<br>
            🛡️ Bundle tech support at reduced price
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
            ✅ Maintain quality service<br>
            🌟 Loyalty reward for long-term customers
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary table
    st.markdown("### 📋 Input Summary")
    summary_df = pd.DataFrame({
        'Feature': ['Tenure', 'Contract', 'Monthly Charges', 'Internet Service', 'Tech Support', 'Payment Method'],
        'Value': [f'{tenure} months', contract, f'${monthly_charges}', internet_service, tech_support, payment_method]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa;'>Made with Streamlit · XGBoost · Telco Dataset</p>", unsafe_allow_html=True)
