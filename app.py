## Install Dependencies
import os
import random
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

import gspread
from oauth2client.service_account import ServiceAccountCredentials


## CREDENTIALS - Custom DEPENDENCIES
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

### Dictionary/Labels
sex_map = {'male': 1.0, 'female': 0.0}
smoker_map = {'yes': 1.0, 'no': 0.0}
region_map = {'southeast': 3.0, 'southwest': 4.0,
              'northwest': 2.0, 'northeast': 1.0}

## Create Page
### Title
html_temp = """
<div>
    <!-- <h1 style="color:#ddd; background-color:#c3c3c3;  text-align:center;
    justify-content: center; align-items: center;">
        <strong>Insurance</strong>
    </h1> -->
    <h2 style="padding-top: 0px; text-align: center;">
        PREDICTION OF MEDICAL PERSONNEL CHARGES
    <h2>
</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)

if st.checkbox("Information"):
    html_support = """
        <h5>
            Please be informed that modeled datasets are in the public domain, sourced from 
            <a href="https://github.com/stedy/Machine-Learning-with-R-datasets"> Machine Learning with R by Brett Lantz. </a> 
            and contains simulated data, on the basis of demographic statistics from the US Census Bureau
        </h5>
        
        <h5>
            An end-to-end predictive web application that reinforces with each submission.
            Enter the application form with the details that best describes you, and choose to contribute your data,
            a copy would be generated for download.
        </h5>
        
        <hr>
    """

    st.markdown(html_support, unsafe_allow_html=True)



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


### Explore
st.subheader("Choose to share")
if st.checkbox("I Agree"):
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
    #
    # CREDENTIALS = {
    #     "type": "service_account",
    #     "project_id": "gs-database-353116",
    #     "private_key_id": "9e98c102718109023497fb78795f323ba6e01cbd",
    #     "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDnVqHpnJHaRaMR\nzYReoGjT0tRFenslmCLHNcfzlleRxZlclBXooHPK2g5Fn4Wx+VGbcJpNigGZ0XMA\nItXZR6rcnYMj7qVp0aHHS/7icqb0Pj0nzogxidS05WUyQZ9fVBj2dqbHF2CXHiOH\n483m7cTgZse55aonwz+BJ+waqe7Fak0fQMVIgNuvc39aQtJ5p1LBy1Xb9+MQhTFL\na5BWaWrhn869Wft7xUgVpIAY5Hr9uz2ssB+DhF1jQwaCvrhNvzQ8PWFiPxW+KUsJ\nhoLzB2YDK5k9M64TfpvZUTzFg+BRiCvYi8ZBkh2NZCfo5YqihKVFvya8os4W3Dd8\ngBHV4U8FAgMBAAECggEALyWgTPicXxQ3Jj8w3YoqoxUCLwDFzUUivV/QFuxKf4p6\ndXw2tNjSkIJ9SyI2QK3TvG6n++qG5f7iRaJT2/r3rwuo+O8/pf/TzUbHNQZx0TBI\nDW9RoWr6Pz3LMIFgRjDg/4Xx+nxgspuxWRYL1icaKzO3O8M7OOxZamyk+VAtuezL\nfhwpTjPbBozoFSJAhw3OVArqsI4R4dUnls/Mym1EYhxrl1twuzhYVwniD2O++XLm\ngb6wwf6cfAV86+80E840fGLqQce4TWOuCi+xO8JlXwu/8OsMkVbVvfW4p0Xk7TAk\nZ9APRlUK2wl3GBUylMqIPbkFL5mVMloWNPGL9ELVZwKBgQD5lomgdjNkkpn9w0cZ\nfS34aeVUDzQHe9Dig/78KYiHrYfdJD7QDNoF47LGxpfcvHjABTaMkwcAS5LVLRBS\nDLqNcQ1Ydkmu4aOQidJ7C1xJNQI97jQJGjPv9KSEevmCdhzlwwBTNYHpoG3gRU3p\nTqTarBjRHWg25HdvleqadL1RQwKBgQDtSBKgy1eNW4HCLkde1uG6N9PwHn8ZMbA+\n8q0I0cmh4r6G9+nhfC5Ncrukp5DpZ2AHDrZoti1y0MC5nvjo5N5tdGwBNBBs3jqC\nmgsdxmvZr0WoE9wl4a4gOr5ShCA8ntdkLT65MwXqkrqI5CFtD21/8OawOvGM25An\natkvKbvWFwKBgQCip55telqn0nqUzCyLye6pk6mmjHnl3qUU2dzUzORzN33xemuM\n/rMfX3Lk5AuYCSPQUBVqq27GnHnGf6XBMxZokyKVYhFG7TnBOnB3S8IK24bogVJc\nFD2AxbhpthLhGMRgYCLYF2jgrnKs072grGX3NGy/6yA9lcrYZ6UKn8W5LQKBgQC5\nTfjmRAcK6PSsHhI0uaGtGR5VvqtJlKlsb368jYFnALoEk6W+J4nNiBWMoCQmc0nm\nGRJRMjzKFsb772+6Ccq/NhWG1w8gxmhxSDX4OdZOOXgvq9rYZqfimZ26uV6nmPDj\nVgZPAc7UA2TTtT15e3vrV8oAxPeRJoMslWApWfFMXwKBgQD5BaagfXRyceyiYSpz\npbEcK8iY0M25y2/ne+dHos+Xz98dsx7HIO1vHQ0WMilUrZvxYgjOK25XaABftDBo\n2rdRSamEed2n0iwcIMj3ZyvmQisDuUH5Pv3u09O9Jl+k0mYIZ+ETRRpiTnvnR4l8\n9LkGEiGm4EllJBzApJI2yu+hwg==\n-----END PRIVATE KEY-----\n",
    #     "client_email": "dan-700@gs-database-353116.iam.gserviceaccount.com",
    #     "client_id": "103885161649534649705",
    #     "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    #     "token_uri": "https://oauth2.googleapis.com/token",
    #     "auth_provider_x509_cert_url ": "https://www.googleapis.com/oauth2/v1/certs",
    #     "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/dan-700%40gs-database-353116.iam.gserviceaccount.com"
    # }

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


## Explore
st.sidebar.subheader("Explore")
st.sidebar.write("Check out [PORTFOLIO](https://akanimohod19a.github.io/)")
st.sidebar.write("Check out demo [APPS](https://heartfailurepredictor-afl.herokuapp.com/)")
