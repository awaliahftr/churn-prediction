🔍 Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

> Predicting which customers will churn using machine learning — with an interactive web app for business teams to use without any coding.

🎯 Problem Statement

A telecom company is losing customers, directly impacting revenue. This project:
- Identifies the key drivers of customer churn
- Builds a predictive ML model to flag at-risk customers
- Delivers an interactive app so business teams can act on predictions


📊 Key Findings

- 📌 Overall churn rate is 26.5% — 1 in 4 customers leave
- 📌 Month-to-month contract customers churn 3× more than annual subscribers
- 📌 Customers with tenure under 12 months are the highest risk group
- 📌 High monthly charges (>$65) correlate strongly with churn
- 📌 Customers without tech support are 2× more likely to churn

🤖 Model Performance

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 80% | 0.84 |
| Random Forest | 83% | 0.87 |
| **XGBoost ✅** | **85%** | **0.90** |

> XGBoost selected as final model — highest accuracy and recall score.

💡 Business Recommendations
| # | Action | Impact |
|---|---|---|
| 1 | Offer discounts to convert month-to-month → annual plans | 🔴 High |
| 2 | Create onboarding program for new customers (first 12 months) | 🔴 High |
| 3 | Bundle tech support with basic plans | 🟡 Medium |
| 4 | Flag high-risk customers monthly using the ML model | 🔴 High |

🛠️ Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Web App | Streamlit |
| Version Control | Git & GitHub |

📁 Project Structure
```
churn-prediction/
├── churn_analysis.ipynb    # Full EDA + ML analysis
├── app.py                  # Streamlit prediction app
├── requirements.txt        # Dependencies
├── model/
│   ├── xgb_churn_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── images/
    ├── churn_overview.png
    ├── roc_curve.png
    └── feature_importance.png
```
🚀 How to Run Locally
```
Clone the repo:
- git clone https://github.com/awaliahftr/churn-prediction.git
- cd churn-prediction

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py
```
📦 Dataset
Source: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Size: 7,043 customers, 21 features
Target: Churn (Yes / No)

👤 Author
Awaliahftr
🔗 LinkedIn: https://www.linkedin.com/in/awaliahftrr
📧 Email: awaliahftrr@gmail.com
