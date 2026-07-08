import streamlit as st
from inference import (
    load_model, load_data, recompute, load_features, get_application_data, predict
)

st.set_page_config(page_title="Home Credit Risk Dashboard", page_icon="💳", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_feature():
    return load_features()


@st.cache_data
def get_data():
    return load_data()


model = get_model()
selected_features = get_feature()
df = get_data()

st.title("Home Credit Default Risk Dashboard")
st.caption("Select a sample applicant, then adjust their declarable details to see how risk changes. "
           "Bureau and credit-history data stay fixed to the sampled applicant's real records.")

st.divider()

sk_id = st.selectbox("Select Applicant ID", df["SK_ID_CURR"].tolist())
row = get_application_data(df, sk_id)

st.subheader("Application Details")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input(
        "Annual Income", value=float(row["AMT_INCOME_TOTAL"].values[0]),
        min_value=0.0, max_value=10_000_000.0
    )
    annuity = st.number_input(
        "Annuity", value=float(row["AMT_ANNUITY"].values[0]),
        min_value=0.0, max_value=1_000_000.0
    )
    family_members = st.slider(
        "Family Members", 1, 10, value=int(row["CNT_FAM_MEMBERS"].values[0])
    )
with col2:
    credit = st.number_input(
        "Credit Amount", value=float(row["AMT_CREDIT"].values[0]),
        min_value=0.0, max_value=10_000_000.0
    )
    good_price = st.number_input(
        "Goods Price", value=float(row["AMT_GOODS_PRICE"].values[0]),
        min_value=0.0, max_value=10_000_000.0
    )
    days_employed = st.number_input(
        "Days Employed (negative = currently employed)",
        value=float(row["DAYS_EMPLOYED"].values[0]),
        min_value=-20_000.0, max_value=0.0
    )

st.divider()

if st.button("Predict Risk", type="primary"):
    with st.spinner("Calculating risk..."):
        row["AMT_INCOME_TOTAL"] = income
        row["AMT_CREDIT"] = credit
        row["AMT_ANNUITY"] = annuity
        row["AMT_GOODS_PRICE"] = good_price
        row["CNT_FAM_MEMBERS"] = family_members
        row["DAYS_EMPLOYED"] = days_employed
        row = recompute(row)
        prob = predict(model, row, selected_features)

    st.toast("Prediction complete", icon="✅")

    st.subheader("Result")
    m1, m2 = st.columns(2)
    m1.metric("Default Probability", f"{prob:.2%}")

    if prob < 0.30:
        m2.metric("Risk Band", "Low")
        st.success(f"Low Risk — Default Probability: {prob:.2%}")
    elif prob < 0.60:
        m2.metric("Risk Band", "Medium")
        st.warning(f"Medium Risk — Default Probability: {prob:.2%}")
    else:
        m2.metric("Risk Band", "High")
        st.error(f"High Risk — Default Probability: {prob:.2%}")