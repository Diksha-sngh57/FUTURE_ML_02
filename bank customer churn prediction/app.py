import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")

# Title and description
st.title("Bank Customer Churn Prediction")
st.markdown("""
This application predicts whether a bank customer is likely to churn based on their profile.
Enter the customer details below to get a prediction.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Churn_Modelling.csv')
    
    # Handle missing values
    numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    categorical_cols = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    # Prepare features and target
    X = df.drop(['Exited', 'Surname', 'RowNumber', 'CustomerId'], axis=1)
    y = df['Exited']
    
    return X, y, df

X, y, df = load_data()

# Train model
@st.cache_resource
def train_model(X, y):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X, y)
    return xgb_model

model = train_model(X, y)

# Sidebar for user input
st.sidebar.header("Customer Information")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 10000.0)
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
geography = st.sidebar.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.checkbox("Has Credit Card")
is_active_member = st.sidebar.checkbox("Is Active Member")

# Prepare input data for prediction
input_data = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if geography == 'France' else 0,
    'Geography_Germany': 1 if geography == 'Germany' else 0,
    'Geography_Spain': 1 if geography == 'Spain' else 0,
    'Gender_Female': 1 if gender == 'Female' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'NumOfProducts_1': 1 if num_products == 1 else 0,
    'NumOfProducts_2': 1 if num_products == 2 else 0,
    'NumOfProducts_3': 1 if num_products == 3 else 0,
    'NumOfProducts_4': 1 if num_products == 4 else 0,
    'HasCrCard_0': 0 if has_cr_card else 1,
    'HasCrCard_1': 1 if has_cr_card else 0,
    'IsActiveMember_0': 0 if is_active_member else 1,
    'IsActiveMember_1': 1 if is_active_member else 0
}

input_df = pd.DataFrame([input_data])

# Ensure input_df columns match X columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"The customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"The customer is unlikely to churn (Probability: {probability:.2%})")

# Data Visualizations
st.subheader("Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.write("Distribution of Exited Customers")
    fig, ax = plt.subplots()
    sns.countplot(x='Exited', data=df)
    ax.set_title('Distribution of Exited Customers')
    ax.set_xlabel('Exited')
    ax.set_ylabel('Count')
    st.pyplot(fig)

with col2:
    st.write("Credit Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['CreditScore'], kde=True)
    ax.set_title('Distribution of CreditScore')
    ax.set_xlabel('CreditScore')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Correlation Matrix
st.subheader("Correlation Matrix")
numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
correlation_matrix = df[numerical_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
ax.set_title('Correlation Matrix of Numerical Features')
st.pyplot(fig)