## Install Dependencies

import os
import random
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from _datetime import datetime

import gspread
from oauth2client.service_account import ServiceAccountCredentials


## CREDENTIALS
# => Update Google Sheet
def update_sheet(client):
    sheet = client.open('INSURANCE')
    sheet_insurance = sheet.get_worksheet(0)
    sheet_insurance.insert_rows(df.values.tolist())


# => Load Model
def load_model(modelfile):
    model = pickle.load(open(modelfile, "rb"))
    return model


# => Retrievng Values
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# => Retrievng Values II
def get_fvalue(val):
    feature_dict = {"yes": 1, "no": 0}
    for key, value in feature_dict.items():
        if val == key:
            return value


## Create Page
### Title
html_temp = """
<div>
    <h1 style="color:#ddd; background-color:#c3c3c3;  text-align:center;
    justify-content: center; align-items: center;">
        <strong>Insurance</strong>
    </h1>
    <h2 style="padding-top: 0px; text-align: center;">
        Prediction of Medical Personnel Charges
    <h2>
</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)

### Dictionary/Labels
sex_map = {'male': 1.0, 'female': 0.0}
smoker_map = {'yes': 1.0, 'no': 0.0}
region_map = {'southeast': 3.0, 'southwest': 4.0,
              'northwest': 2.0, 'northeast': 1.0}

### Prediction Form
st.sidebar.title("Prediction Form")

### Inputs
# Name
r_id = random.randint(1000, 1999)
name = "User" + str(r_id)
# Age
age = st.sidebar.number_input("Age", 1, 100, 18, 1)
# Sex
sex = st.sidebar.radio("Sex", tuple(sex_map.keys()))
# Region
region = st.sidebar.radio("Region", tuple(region_map.keys()))
# Body Mass Index - min, max, val, step
bmi = st.sidebar.number_input("Body Mass Index", 10.00, 100.00, 15.96, 0.10)
# Children
children = st.sidebar.number_input("Number of children", min_value=0, value=1, step=1)
# Smoker
smoker = st.sidebar.radio("Do you Smoke?", tuple(smoker_map.keys()))
# Time
entry_date = datetime.now().strftime("%d-%m-%Y")
entry_dir = 'contrib' + '_' + entry_date

## Feature Values
feature_values = [age, get_value(sex, sex_map),
                  bmi, children, get_fvalue(smoker),
                  get_value(region, region_map)]

### Collect
pretty_results = {'age': [age], 'sex': [sex],
                  'bmi': [bmi], 'children': [children],
                  'smoker': [smoker], 'region': [region]}

print_results = {'age': age, 'sex': sex,
                 'bmi': bmi, 'children': children,
                 'smoker': smoker, 'region': region}

### Reshaped Values
single_sample = np.array(feature_values).reshape(1, -1)

## Make and Fetch Prediction
model = load_model('model.pk_Charges')
prediction = model.predict(single_sample)

pretty_results["p_CHARGES"] = prediction[0]
prediction_table = pd.DataFrame(pretty_results, index=["Proba"])

# => convert dict to df
df = pd.DataFrame(pretty_results)
df['entry'] = entry_date
df['name'] = name

### Explore
st.subheader("Choose to share")
if st.checkbox("Yes"):
    """
    ### Entered Values
    """

    CREDENTIALS = {
        "type": "service_account",
        "project_id": "gs-database-353116",
        "private_key_id": os.environ["private_key_id"],
        "private_key": os.environ["private_key"],
        "client_email": os.environ["client_email"],
        "client_id": os.environ["client_id"],
        "auth_url": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.environ["client_x509_cert_url"]
    }

    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(CREDENTIALS, scope)

    client = gspread.authorize(credentials)
    print("Sending credentials")

    ### Create Paths
    contrib_path = "contrib" + "/" + str(entry_dir)
    csv_path = str(contrib_path) + "/" + str(name) + ".csv"
    st.json(print_results)
    if not os.path.exists(contrib_path):
        os.makedirs(contrib_path)
        df.to_csv(csv_path, index=False)
        # - Update sheet
        update_sheet(client)
    elif os.path.exists(contrib_path):
        df.to_csv(csv_path, index=False)
        # - Update sheet
        update_sheet(client)
    ## Download Copy
    with open(csv_path, "rb") as file:
        btn = st.download_button(
            label="Download a copy",
            data=file,
            file_name="Insurance-contrib.csv",
            mime="application/octet-stream"
        )
        st.write('Thanks for contributing!')

"""
### Likelihood of Charges
"""

if st.button("Predict"):
    '''
    ## Results
    '''
    st.caption("Prediction Table")
    st.table(prediction_table)
    st.warning("[+-] From our model the Charge would a variance of (give or take) 3981.35")

## Explore
st.sidebar.subheader("Explore")
