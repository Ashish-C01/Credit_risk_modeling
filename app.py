import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.sav', 'rb'))

st.title("Credit Risk Modeling")
st.write('A machine learning model to segregate customers based on their previous financial information for loan approval. The model determines the risk of lending money to a customer by assigning it to a specific class.')
pct_tl_open_L6M = int(st.number_input(
    "Percent accounts opened in last 6 months"))
pct_tl_closed_L6M = st.number_input("percent accounts closed in last 6 months")
Tot_TL_closed_L12M = st.number_input(
    "Total accounts closed in last 12 months", min_value=0, step=1)
pct_tl_closed_L12M = st.number_input(
    "Percent accounts closed in last 12 months")
Tot_Missed_Pmnt = st.number_input("Total missed payments", min_value=0, step=1)
CC_TL = st.number_input("Count of Credit card accounts", min_value=0, step=1)
Home_TL = st.number_input(
    "Count of Housing loan accounts", min_value=0, step=1)
PL_TL = st.number_input("Count of Personal loan accounts", min_value=0, step=1)
Secured_TL = st.number_input("Count of secured accounts", min_value=0, step=1)
Unsecured_TL = st.number_input(
    "Count of unsecured accounts", min_value=0, step=1)
Other_TL = st.number_input("Count of other accounts", min_value=0, step=1)
Age_Oldest_TL = st.number_input(
    "Age of oldest opened account", min_value=0, step=1)
Age_Newest_TL = st.number_input(
    "Age of newest opened account", min_value=0, step=1)
time_since_recent_payment = st.number_input(
    "Time Since recent Payment made", min_value=0, step=1)
max_recent_level_of_deliq = st.number_input(
    "Maximum recent level of delinquency", min_value=0, step=1)
num_deliq_6_12mts = st.number_input(
    "Number of times delinquent between last 6 months and last 12 months", min_value=0, step=1)
num_times_60p_dpd = st.number_input(
    "Number of times 60+ dpd", min_value=0, step=1)
num_std_12mts = st.number_input(
    "Number of standard Payments in last 12 months", min_value=0, step=1)


num_sub = st.number_input(
    "Number of sub standard payments - not making full payments", min_value=0, step=1)
num_sub_6mts = st.number_input(
    "Number of sub standard payments in last 6 months", min_value=0, step=1)
num_sub_12mts = st.number_input(
    "Number of sub standard payments in last 12 months", min_value=0, step=1)
num_dbt = st.number_input("Number of doubtful payments", min_value=0, step=1)
num_dbt_12mts = st.number_input(
    "Number of doubtful payments in last 12 months", min_value=0, step=1)
num_lss = st.number_input("Number of loss accounts ", min_value=0, step=1)
recent_level_of_deliq = st.number_input(
    "Recent level of delinquency", min_value=0, step=1)
CC_enq_L12m = st.number_input(
    "Credit card enquiries in last 12 months", min_value=0, step=1)
PL_enq_L12m = st.number_input(
    "Personal Loan enquiries in last 12 months", min_value=0, step=1)
time_since_recent_enq = st.number_input(
    "Time since recent enquiry", min_value=0, step=1)
enq_L3m = st.number_input("Enquiries in last 3 months", min_value=0, step=1)
NETMONTHLYINCOME = st.number_input("Net Monthly Income", min_value=0, step=1)
Time_With_Curr_Empr = st.number_input(
    "Time with current Employer", min_value=0, step=1)


pct_PL_enq_L6m_of_ever = st.number_input(
    "Percent enquiries PL in last 6 months to last 6 months")
pct_CC_enq_L6m_of_ever = st.number_input(
    "Percent enquiries CC in last 6 months to last 6 months")


CC_Flag = st.radio('Credit card Flag', [0, 1])
PL_Flag = st.radio('Personal Loan Flag', [0, 1])
HL_Flag = st.radio('Housing loan flag', [0, 1])
GL_Flag = st.radio('Gold loan flag', [0, 1])
education = st.radio('Education', ['SSC', '12TH', 'GRADUATE',
                     'UNDER GRADUATE', 'POST-GRADUATE', 'PROFESSIONAL', 'OTHERS'])
marital_status = st.radio('Marital Status', ['Married', 'Single'])
gender = st.radio('Gender', ['Male', 'Female'])
last_prod_enq2 = st.radio('Latest product enquired for', [
                          'AL', 'Credit Card', 'Consumer Loan', 'Housing Loan', 'Personal Loan', 'Others'])
first_prod_enq2 = st.radio('First product enquired for', [
                           'AL', 'Credit Card', 'Consumer Loan', 'Housing Loan', 'Personal Loan', 'Others'])


if st.button('Enter'):
    features = []
    features.extend([pct_tl_open_L6M, pct_tl_closed_L6M, Tot_TL_closed_L12M, pct_tl_closed_L6M, Tot_Missed_Pmnt, CC_TL,
                    Home_TL, PL_TL, Secured_TL, Unsecured_TL, Other_TL, Age_Oldest_TL, Age_Newest_TL, time_since_recent_payment, max_recent_level_of_deliq, num_deliq_6_12mts, num_times_60p_dpd, num_std_12mts, num_sub, num_sub_6mts, num_sub_12mts, num_dbt, num_dbt_12mts, num_lss, recent_level_of_deliq, CC_enq_L12m, PL_enq_L12m, time_since_recent_enq, enq_L3m, NETMONTHLYINCOME, Time_With_Curr_Empr, CC_Flag, PL_Flag, pct_PL_enq_L6m_of_ever, pct_CC_enq_L6m_of_ever, HL_Flag, GL_Flag])
    if education == 'SSC':
        features.append(1)
    elif education == 'OTHERS':
        features.append(1)
    elif education == '12TH':
        features.append(2)
    elif education == 'GRADUATE':
        features.append(3)
    elif education == 'UNDER GRADUATE':
        features.append(3)
    elif education == 'POST-GRADUATE':
        features.append(4)
    elif education == 'PROFESSIONAL':
        features.append(3)

    # For marital status
    if marital_status == 'Married':
        features.extend([True, False])
    else:
        features.extend([False, True])

    # For gender
    if gender == 'Female':
        features.extend([True, False])
    else:
        features.extend([False, True])

    # For latest product enquiry
    if last_prod_enq2 == 'AL':
        features.extend([True, False, False, False, False, False])
    elif last_prod_enq2 == 'Credit Card':
        features.extend([False, True, False, False, False, False])
    elif last_prod_enq2 == 'Consumer Loan':
        features.extend([False, False, True, False, False, False])
    elif last_prod_enq2 == 'Housing Loan':
        features.extend([False, False, False, True, False, False])
    elif last_prod_enq2 == 'Personal Loan':
        features.extend([False, False, False, False, True, False])
    elif last_prod_enq2 == 'Others':
        features.extend([False, False, False, False, False, True])

    # For first product inquiry
    if first_prod_enq2 == 'AL':
        features.extend([
                        True, False, False, False, False, False])
    elif first_prod_enq2 == 'Credit Card':
        features.extend([
                        False, True, False, False, False, False])
    elif first_prod_enq2 == 'Consumer Loan':
        features.extend([
                        False, False, True, False, False, False])
    elif first_prod_enq2 == 'Housing Loan':
        features.extend([
                        False, False, False, True, False, False])
    elif first_prod_enq2 == 'Personal Loan':
        features.extend([
                        False, False, False, False, True, False])
    elif first_prod_enq2 == 'Others':
        features.extend([
                        False, False, False, False, False, True])
    input_values = np.array([features])
    print("Enter button pressed")
    print(input_values)
    p_class = ['P1', 'P2', 'P3', 'P4']
    prediction = model.predict(input_values)[0]
    st.write(f'Predicted class:  {p_class[prediction]}')
    st.dataframe(pd.DataFrame(
        {'Class': ['P1', 'P2', 'P3', 'P4'], 'Priority Levels for giving loan (higher is better)': [4, 3, 2, 1]}), hide_index=True)
