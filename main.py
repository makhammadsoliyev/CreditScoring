import os
import streamlit as st
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder

# Load the trained model
model = load_model('scoring_model.h5')

# Define the encoders and scaler
payment_behaviour_categories = ['Low_spent_Small_value_payments', 'Low_spent_Medium_value_payments',
                                'Low_spent_Large_value_payments', 'High_spent_Small_value_payments',
                                'High_spent_Medium_value_payments', 'High_spent_Large_value_payments']
payment_behaviour_encoder = OrdinalEncoder(categories=[payment_behaviour_categories])
credit_mix_encoder = OrdinalEncoder(categories=[['Good', 'Standard', 'Bad']])
min_amount_encoder = LabelEncoder()
occupation_encoder = LabelEncoder()

# Sample data to fit the encoders (replace this with actual training data)
sample_data = pd.DataFrame({
    'Payment_Behaviour': ['Low_spent_Small_value_payments', 'High_spent_Large_value_payments'],
    'Credit_Mix': ['Good', 'Bad'],
    'Payment_of_Min_Amount': ['Yes', 'No'],
    'Occupation': ['Scientist', 'Teacher']
})
payment_behaviour_encoder.fit(sample_data[['Payment_Behaviour']])
credit_mix_encoder.fit(sample_data[['Credit_Mix']])
min_amount_encoder.fit(sample_data['Payment_of_Min_Amount'])
occupation_encoder.fit(['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Lawyer', 'Developer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])

# Create a sample data for the MinMaxScaler fitting
combined_sample_data = pd.DataFrame({
    'Age': [23, 23],
    'Occupation': occupation_encoder.transform(['Scientist', 'Scientist']),
    'Annual_Income': [19114.12, 19114.12],
    'Monthly_Inhand_Salary': [1824.8433333333328, 1824.8433333333328],
    'Num_Bank_Accounts': [3, 3],
    'Num_Credit_Card': [4, 4],
    'Interest_Rate': [3.0, 3.0],
    'Num_of_Loan': [4.0, 4.0],
    'Delay_from_due_date': [3, -1],
    'Num_of_Delayed_Payment': [7.0, 0.0],
    'Changed_Credit_Limit': [11.27, 11.27],
    'Num_Credit_Inquiries': [4.0, 4.0],
    'Credit_Mix': credit_mix_encoder.transform([['Good'], ['Good']]).flatten(),
    'Outstanding_Debt': [809.98, 809.98],
    'Credit_Utilization_Ratio': [26.82, 31.94],
    'Payment_of_Min_Amount': min_amount_encoder.transform(['No', 'No']),
    'Total_EMI_per_month': [49.57, 49.57],
    'Amount_invested_monthly': [118.28, 118.28],
    'Payment_Behaviour': payment_behaviour_encoder.transform([['High_spent_Small_value_payments'], ['Low_spent_Large_value_payments']]).flatten(),
    'Monthly_Balance': [312.49, 284.63],
    'Credit_History_Age_Months': [265.0, 265.0]
})

scaler = MinMaxScaler()
scaler.fit(combined_sample_data)

# Function to encode inputs
def encode_inputs(df):
    df['Payment_Behaviour'] = payment_behaviour_encoder.transform(df[['Payment_Behaviour']])
    df['Credit_Mix'] = credit_mix_encoder.transform(df[['Credit_Mix']])
    df['Payment_of_Min_Amount'] = min_amount_encoder.transform(df['Payment_of_Min_Amount'])
    df['Occupation'] = occupation_encoder.transform(df['Occupation'])
    return df

# Streamlit application
st.title("Kredit skoring")

# Input features
age = st.number_input('Yosh', min_value=0, max_value=100, value=23)
occupation = st.selectbox('Kasb', ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Lawyer', 'Developer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])
annual_income = st.number_input('Yillik daromad', min_value=0.0, max_value=1e6, value=19114.12)
monthly_inhand_salary = st.number_input('Oylik daromad', min_value=0.0, max_value=1e5, value=1824.84)
num_bank_accounts = st.number_input('Bank hisoblari soni', min_value=0, max_value=50, value=3)
num_credit_cards = st.number_input('Kredit kartalar soni', min_value=0, max_value=50, value=4)
interest_rate = st.number_input('Stavka foizi', min_value=0.0, max_value=100.0, value=3.0)
num_of_loan = st.number_input('Kreditlar soni', min_value=0.0, max_value=10.0, value=4.0)
delay_from_due_date = st.number_input('Tugatish sanasidan kechikish', min_value=-10, max_value=50, value=3)
num_of_delayed_payment = st.number_input('Kechiktirilgan to\'lovlar soni', min_value=0.0, max_value=20.0, value=7.0)
changed_credit_limit = st.number_input('O\'zgartirilgan kredit limiti', min_value=-5000.0, max_value=5000.0, value=11.27)
num_credit_inquiries = st.number_input('Kredit so\'rovlari soni', min_value=0.0, max_value=20.0, value=4.0)
credit_mix = st.selectbox('Kredit aralashmasi', ['Good', 'Standard', 'Bad'])
outstanding_debt = st.number_input('To\'lanmagan qarz', min_value=0.0, max_value=1e6, value=809.98)
credit_utilization_ratio = st.number_input('Kreditdan foydalanish koeffitsienti', min_value=0.0, max_value=100.0, value=26.82)
payment_of_min_amount = st.selectbox('Minimal miqdorni to\'lash', ['Yes', 'No'])
total_emi_per_month = st.number_input('Oyiga jami tenglashtirilgan oylik to\'lov', min_value=0.0, max_value=1e5, value=49.57)
amount_invested_monthly = st.number_input('Oylik investitsiya miqdori', min_value=0.0, max_value=1e4, value=118.28)
payment_behaviour = st.selectbox('To\'lov harakati', payment_behaviour_categories)
monthly_balance = st.number_input('Oylik balans', min_value=-1e4, max_value=1e4, value=312.49)
credit_history_age_months = st.number_input('Kredit tarixi (oylar)', min_value=0.0, max_value=1000.0, value=26.0)

# Create DataFrame from input
input_data = pd.DataFrame({
    'Age': [age],
    'Occupation': [occupation],
    'Annual_Income': [annual_income],
    'Monthly_Inhand_Salary': [monthly_inhand_salary],
    'Num_Bank_Accounts': [num_bank_accounts],
    'Num_Credit_Card': [num_credit_cards],
    'Interest_Rate': [interest_rate],
    'Num_of_Loan': [num_of_loan],
    'Delay_from_due_date': [delay_from_due_date],
    'Num_of_Delayed_Payment': [num_of_delayed_payment],
    'Changed_Credit_Limit': [changed_credit_limit],
    'Num_Credit_Inquiries': [num_credit_inquiries],
    'Credit_Mix': [credit_mix],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_Utilization_Ratio': [credit_utilization_ratio],
    'Payment_of_Min_Amount': [payment_of_min_amount],
    'Total_EMI_per_month': [total_emi_per_month],
    'Amount_invested_monthly': [amount_invested_monthly],
    'Payment_Behaviour': [payment_behaviour],
    'Monthly_Balance': [monthly_balance],
    'Credit_History_Age_Months': [credit_history_age_months]
})

# Encode the input features
input_data = encode_inputs(input_data)

# Normalize the input data
normalized_input = scaler.transform(input_data)
categories=['A\'lo', 'Yaxshi', 'Qoniqarsiz']
# Make prediction
if st.button('Bashorat qilish'):
    prediction = model.predict(normalized_input)
    prediction_class = prediction.argmax(axis=1)
    st.write(f"Natija: {categories[prediction_class[0]-1]}")

