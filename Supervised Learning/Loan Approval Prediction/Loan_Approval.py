#%%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("data.csv")
# %%
# df.head()
df.info()
# %%
df['Married'].unique()

df.drop(['Loan_ID'],axis=1,inplace=True)

df['Gender'] = df['Gender'].map({'Male' : 0 , 'Female':1 } )

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)

df['Married']= df['Married'].map({"Yes":1,"No":0})

df['Married'].fillna(df['Married'].mode()[0],inplace=True)
# %%
# 
df['Dependents'].unique()
df['Dependents']= df['Dependents'].replace('3+',3)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# %%
df['Education'] = df['Education'].map({'Graduate' : 1,'Not Graduate' : 0})


# %%

df['Self_Employed'].unique()
df['Self_Employed'] = df['Self_Employed'].map({'Yes' : 1,'No' : 0})
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)

# %%
df['LoanAmount']= df['LoanAmount'].fillna(df['LoanAmount'].median())
# %%
df['Loan_Amount_Term']= df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
# %%
df['Credit_History'].unique()

df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
# %%
df['Loan_Status'].unique()
df = df.dropna(subset=['Loan_Status'])
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})


# %%

ohe = OneHotEncoder(drop=None, sparse_output=False)


encoded = ohe.fit_transform(df[['Property_Area']])


encoded_df = pd.DataFrame(encoded, 
                          columns=ohe.get_feature_names_out(['Property_Area']),
                          index=df.index)


df = pd.concat([df.drop(columns=['Property_Area']), encoded_df], axis=1)


print("New columns created:")
print([col for col in df.columns if 'Property_Area' in col])
print(f"\nShape after encoding: {df.shape}")
print("\nFirst few rows of encoded columns:")
print(df[[col for col in df.columns if 'Property_Area' in col]].head())
# %%




# %%
iqr_cols = ['ApplicantIncome',	'CoapplicantIncome',	'LoanAmount'	,'Loan_Amount_Term']
df_iqr = df.copy()

# %%
for col in iqr_cols:
    q1 = df_iqr[col].quantile(0.25)
    q3 = df_iqr[col].quantile(0.75)
    iqr = q3-q1
    lower = q1- (1.5*iqr)
    upper = q3+ (1.5*iqr)
    df_iqr = df_iqr[(df_iqr[col] >= lower) & (df_iqr[col] <= upper)]


# log reg
X = df_iqr.drop(["Loan_Status"], axis=1)
y = df_iqr['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)




############################################################
# streamlit
st.set_page_config(page_title="Loan Prediction App", layout="wide")
st.title("💳 Loan Approval Prediction")
st.markdown("Use this app to predict whether a loan will be approved based on applicant details.")


st.sidebar.header("📋 Enter Applicant Details")

with st.sidebar.expander("Basic Info"):
    gender = st.selectbox("Gender", options=["Male", "Female"], help="Select the applicant's gender")
    married = st.radio("Married", options=["Yes", "No"], help="Is the applicant married?")
    dependents = st.selectbox("Dependents", options=[0, 1, 2, 3], help="Number of dependents")
    education = st.radio("Education", options=["Graduate", "Not Graduate"], help="Education level")
    self_employed = st.radio("Self Employed", options=["Yes", "No"], help="Is the applicant self-employed?")
    property_area = st.selectbox("Property Area", options=["Urban", "Rural", "Semiurban"], help="Location type")

with st.sidebar.expander("Financial Info"):
    applicant_income = st.slider("Applicant Income", min_value=0, max_value=100000, value=5000, step=500)
    coapplicant_income = st.slider("Coapplicant Income", min_value=0, max_value=50000, value=2000, step=500)
    loan_amount = st.slider("Loan Amount (in thousands)", min_value=0, max_value=700, value=150, step=10)
    loan_amount_term = st.slider("Loan Amount Term (in days)", min_value=0, max_value=500, value=360, step=10)
    credit_history = st.radio("Credit History", options=[1.0, 0.0], help="1 = good credit, 0 = bad credit")


input_dict = {
    'Gender': 0 if gender == "Male" else 1,
    'Married': 1 if married == "Yes" else 0,
    'Dependents': dependents,
    'Education': 1 if education == "Graduate" else 0,
    'Self_Employed': 1 if self_employed == "Yes" else 0,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area_Rural': 1 if property_area == "Rural" else 0,
    'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
    'Property_Area_Urban': 1 if property_area == "Urban" else 0
}
input_df = pd.DataFrame([input_dict])


input_df = input_df.fillna(0)


prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]


st.subheader("📊 Prediction Result")
if prediction == 1:
    st.success("✅ Loan Approved")
else:
    st.error("❌ Loan Not Approved")


fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={'text': "Approval Probability"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "green" if prediction == 1 else "red"}}
))
st.plotly_chart(fig, use_container_width=True)


accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: **{accuracy:.2f}**")
