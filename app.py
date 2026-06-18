import streamlit as st
import pandas as pd
import joblib

model = joblib.load('invoice_model.pkl')

st.title('Invoice Payment Prediction')

invoice_currency = st.selectbox('Invoice Currency', ['USD', 'CAD'])
business_code = st.selectbox(
    'Business Code',
    ['U001', 'CA02', 'U013', 'U002', 'U007', 'U005']
)
document_type = st.selectbox(
    'Document type',
    ['RV', 'X2']
)
customer_payment_terms = st.selectbox(
    'Payment Terms',
    ['NAH4', 'NAD1', 'NAA8', 'CA10', 'NAC6', 'NAM2', 'NAAX', 'NAVE', 'NAG2', 'NABG',
     'NAM4', 'NA10', 'NAU5', 'NA32', 'NAD5', 'NAWP', 'NAGD', 'NAVR', 'CA30', 'NAM1',
     'NAAW', 'NAVF', 'NAD4', 'NAUZ', 'NA3F', 'NAX2', 'NAVQ', 'NATM', 'CAB1', 'NA84',
     'NAWM', 'NACB', 'NACG', 'NA38', 'NAWN', 'C106', 'NAWU', 'NAB1', 'NA3B', 'NA9X',
     'NAVD', 'NAVM', 'NACE', 'NA25', 'NAUP', 'NAM3', 'NACH', 'CAX2', 'NATV', 'NAVL',
     'NATZ', 'C129', 'BR56', 'NA31', 'NATW', 'B052', 'NAV2', 'NATX', 'NAUY', 'NA8Q',
     'NATJ', 'BR12', 'NATU', '90M7', 'NAV9', 'NATK', 'CA60', 'NATL', 'NAD8', 'NAUW',
     'NAVC', 'NABD', 'NATH', 'MC15']
)

business_year = st.number_input('Business Year', min_value = 2000, max_value = 2026, value=2000)

total_open_amount = st.number_input('Open Amount', min_value = 0.00, max_value = 999999999.00, value=0.0, step=0.01)

arrears = st.number_input('Arrears', min_value=-999, max_value = 999, value=0)

if st.button('Predict'):
    input_df = pd.DataFrame({
    'business_code': [business_code],
    'buisness_year': [business_year],
    'invoice_currency': [invoice_currency],
    'document type': [document_type],
    'total_open_amount': [total_open_amount],
    'cust_payment_terms': [customer_payment_terms],
    'arrears': [arrears]
})
    probs = model.predict_proba(input_df)[:,1]
    pred = (probs >= 0.76).astype(int)
    if pred[0] == 1:
        st.write('Unpaid - Flag for collection follow-up')
    else:
        st.write('Paid - no action required')