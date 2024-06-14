import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import io

@st.cache
def load_data():
    df = pd.read_csv('transactions.csv')
    return df

def preprocess_data(data, fit_scaler=False, scaler=None):
    # One-hot encode categorical features
    data_encoded = pd.get_dummies(data, columns=['transaction_type'])
    
    if fit_scaler:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(data_encoded)
        return data_encoded, scaler, features_scaled
    else:
        features_scaled = scaler.transform(data_encoded)
        return data_encoded, scaler, features_scaled

def train_model(data):
    # Preprocess data
    data_encoded, scaler, features_scaled = preprocess_data(data, fit_scaler=True)
    target = data['is_fraud']
    
    # Train Isolation Forest model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features_scaled)
    
    # Save the model
    joblib.dump((model, scaler, data_encoded.columns), 'fraud_model.pkl')
    return model, scaler

def predict_fraud(model, scaler, input_data, feature_columns):
    # Ensure all feature columns are present
    input_data = pd.get_dummies(input_data)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_columns]
    
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# Streamlit UI
st.title("Real-Time Fraud Detection System")

# Dataset upload section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")
else:
    data = load_data()
    st.sidebar.info("Using default dataset")

if st.sidebar.button('Train Model'):
    model, scaler = train_model(data)
    st.sidebar.success("Model trained and saved successfully!")

# Load trained model
model, scaler, feature_columns = joblib.load('fraud_model.pkl')

# Real-time transaction input
st.header("Enter Transaction Details")
amount = st.number_input('Amount', min_value=0.0)
transaction_type = st.selectbox('Transaction Type', ['Type1', 'Type2'])
account_age = st.number_input('Account Age', min_value=0)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'amount': [amount],
    'transaction_type': [transaction_type],
    'account_age': [account_age]
})

if st.button('Predict Fraud'):
    prediction = predict_fraud(model, scaler, input_data, feature_columns)
    if prediction == -1:
        st.error("Fraudulent transaction detected!")
    else:
        st.success("Transaction is legitimate.")

# Display flagged transactions
st.header("Flagged Transactions")
flagged_data = data[data['is_fraud'] == 1]
st.write(flagged_data)
