import streamlit as st 
from inference import (
     load_model, load_data,recompute,load_features,get_application_data,predict
)
st.title("House credit default risk loan")
@st.cache_resource
def get_model():
    return load_model()
@st.cache_resource
def get_feature():
    return load_features()
@st.cache_data
def get_data():
    return load_data()
model=get_model()
selected_features=get_feature()
df=get_data()
sk_id = st.selectbox("Select Applicant ID", df["SK_ID_CURR"].tolist())
row = get_application_data(df, sk_id)
st.subheader("APPLICATION Details")
income = st.number_input("Annual Income", value=float(row["AMT_INCOME_TOTAL"].values[0]))
credit=st.number_input("Credit=",value=float(row["AMT_CREDIT"].values[0]))
annuity=st.number_input("Annuity=",value=float(row["AMT_ANNUITY"].values[0]))
good_price=st.number_input("Good_price",value=float(row["AMT_GOODS_PRICE"].values[0]))
family_members=st.slider("Family members",1,10,value=int(row["CNT_FAM_MEMBERS"].values[0]))
days_employed = st.number_input("Days Employed (negative = currently employed)", value=float(row["DAYS_EMPLOYED"].values[0]))

if st.button("Predict"):
    row["AMT_INCOME_TOTAL"]=income
    row["AMT_CREDIT"]=credit
    row["AMT_ANNUITY"]=annuity
    row["AMT_GOODS_PRICE"]=good_price
    row["CNT_FAM_MEMBERS"]=family_members
    row["DAYS_EMPLOYED"]=days_employed
    row=recompute(row)
    prob = predict(model, row, selected_features)
    if prob < 0.30:
        st.success(f"Low Risk — Default Probability: {prob:.2%}")
    elif prob < 0.60:
        st.warning(f"Medium Risk — Default Probability: {prob:.2%}")
    else:
        st.error(f"High Risk — Default Probability: {prob:.2%}")