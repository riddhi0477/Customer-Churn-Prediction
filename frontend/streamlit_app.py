import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Churn Dashboard", layout="wide")

st.title("🚀 AI Customer Churn Prediction System")

st.sidebar.header("📥 Enter Customer Details")

# -------------------------
# CATEGORICAL INPUTS
# -------------------------

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
senior = 1 if senior == "Yes" else 0

partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
partner = 1 if partner == "Yes" else 0

dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
dependents = 1 if dependents == "Yes" else 0

phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
phone = 1 if phone == "Yes" else 0

multiple = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
multiple = {"No":0, "Yes":1, "No phone service":2}[multiple]

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
internet = {"DSL":0, "Fiber optic":1, "No":2}[internet]

online_sec = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_sec = {"No":0, "Yes":1, "No internet service":2}[online_sec]

backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
backup = {"No":0, "Yes":1, "No internet service":2}[backup]

device = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
device = {"No":0, "Yes":1, "No internet service":2}[device]

tech = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
tech = {"No":0, "Yes":1, "No internet service":2}[tech]

tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
tv = {"No":0, "Yes":1, "No internet service":2}[tv]

movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
movies = {"No":0, "Yes":1, "No internet service":2}[movies]

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
contract = {"Month-to-month":0, "One year":1, "Two year":2}[contract]

paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
paperless = 1 if paperless == "Yes" else 0

payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])
payment = {
    "Electronic check":0,
    "Mailed check":1,
    "Bank transfer":2,
    "Credit card":3
}[payment]

# -------------------------
# NUMERICAL INPUTS
# -------------------------

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 50.0)
total = st.sidebar.slider("Total Charges", 0.0, 10000.0, 1000.0)

# -------------------------
# FEATURE ORDER (VERY IMPORTANT)
# -------------------------

features = [
    gender, senior, partner, dependents, tenure,
    phone, multiple, internet, online_sec, backup,
    device, tech, tv, movies, contract,
    paperless, payment, monthly, total
]

# -------------------------
# PREDICT BUTTON
# -------------------------

if st.sidebar.button("Predict"):

    res = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"features": features}
    )

    data = res.json()

    st.subheader("🔍 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("ML Prediction", "Churn" if data["ml"]==1 else "No Churn")
        st.metric("DL Model", data["dl"])

    with col2:
        st.metric("RNN Model", data["rnn"])
        st.metric("LSTM Model", data["lstm"])

    # -------------------------
    # GRAPH
    # -------------------------
    st.subheader("📊 Model Comparison")

    fig, ax = plt.subplots()
    ax.bar(["ML", "DL", "RNN", "LSTM"],
           [data["ml"], data["dl"], data["rnn"], data["lstm"]])
    st.pyplot(fig)

    # -------------------------
    # XAI
    # -------------------------
    st.subheader("🧠 Feature Importance (Explainable AI)")
    st.bar_chart(data["importance"])