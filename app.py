## Install Dependencies
import os
import random
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

##  Data Dashboard
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
import plotly.graph_objects as go


## Update Sheet
import gspread
from oauth2client.service_account import ServiceAccountCredentials

## Cache
# @st.cache()
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

# PAGE CONFIG
st.set_page_config(layout='wide')

### Dictionary/Labels
sex_map = {'male': 1.0, 'female': 0.0}
smoker_map = {'yes': 1.0, 'no': 0.0}
region_map = {'southeast': 3.0, 'southwest': 4.0,
              'northwest': 2.0, 'northeast': 1.0}

## CREDENTIALS - Custom DEPENDENCIES
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

## States
app_mode = st.sidebar.selectbox('Application Mode',
                                ['Home Page', 'Data Dashboard', 'Prediction Page']
                                )
if app_mode == 'Home Page':
    ## Create Page
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

if app_mode == 'Prediction Page':
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
        print("Sending credentials")

        ### Create Paths
        contrib_path = "contrib" + "/" + str(entry_dir)
        csv_path = str(contrib_path) + "/" + str(name) + ".csv"
        st.json(print_results)
        if os.path.exists(contrib_path):
            df.to_csv(csv_path, index=False)
            update_sheet(client)
        elif not os.path.exists(contrib_path):
            os.makedirs(contrib_path)
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
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

if app_mode == 'Data Dashboard':
    warnings.filterwarnings('ignore')


    ## Other Dependencies
    def evaluate_model(model, X_test, y_test, modelName, DataImb):
        print('------------------------------------------------')
        print("Model ", modelName, end="\n")
        print("Data Balancing Type ", DataImb)
        ### Model must be ran outside the function
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print("R2 Score", r2)
        print("RMSE", rmse)
        return [modelName, DataImb, r2, rmse]


    ## Fetch Data
    ### Connect to Google Sheet
    # Get the instance of the Spreadsheet
    sheet = client.open('INSURANCE')

    # Get the first sheet of the Spreadsheet - where we have the data
    sheet_instance = sheet.get_worksheet(0)
    records_data = sheet_instance.get_all_records()  ## json

    # Convert the json to dataframe
    retf_df = pd.DataFrame.from_dict(records_data)
    ## Rename columns
    retf_df.columns = ['age', 'sex', 'bmi',
                  'children', 'smoker',
                  'region', 'charges', 'entry_date', 'user_id']

    retf_df.drop(columns=['entry_date', 'user_id'], inplace=True)

    ## Feature Engineering

    sex_map = {'male': 1, 'female': 0}
    retf_df['sex'] = retf_df['sex'].map(sex_map).astype('int64')

    smoker_map = {'yes': 1, 'no': 0}
    retf_df['smoker'] = retf_df['smoker'].map(smoker_map).astype('int64')

    LE = LabelEncoder()
    retf_df['region'] = LE.fit_transform(retf_df['region'])

    X = retf_df.drop(columns='charges').values
    y = retf_df['charges'].values

    ## Dashboard
    Features, Samples = st.columns(2)

    with Features:
        st.markdown('**FEATURES**')
        Features_text = st.markdown("0")
        rows = retf_df.shape[0]
    with Samples:
        st.markdown('**SAMPLES**')
        Samples_text = st.markdown("0")
        columns = retf_df.shape[1]

    st.markdown("<hr/>", unsafe_allow_html=True)
    Features_text.write(f"<h1 style='text-align: left; color: red;'>{int(rows)}</h1>", unsafe_allow_html=True)
    Samples_text.write(f"<h1 style='text-align: left; color: red;'>{int(columns)}</h1>", unsafe_allow_html=True)

    st.dataframe(retf_df.describe())
    # dimensions = st.radio("What Dimension Do You Want to Show?", ("Rows", "Columns"))
    # if dimensions == "Rows":
    #     st.text("Showing Length of Rows")
    #
    # if dimensions == "Columns":
    #     st.text("Showing Length of Columns")
    #     retf_df.shape[1]

    ### CHARGES
    insurance_charges = retf_df[['charges']].value_counts().to_frame().reset_index().rename(columns={0:'counts'})
    insurance_charges['charges'] = insurance_charges['charges'].apply(lambda x : str(x) + ' ' + "insurance")
    labels = list(insurance_charges['charges'])

    data_table = retf_df[['age', 'sex', 'bmi', 'children', 'smoker', 'region','charges']]
    data_table = data_table.sort_values(['age', 'sex', 'bmi', 'children', 'smoker', 'region','charges'],
                                        ascending=False)[:20]

    fig3 = go.Figure(data=[go.Table(
        header=dict(values=list(data_table.columns),
                    #line_color='darkslategreen',
                    fill_color='lightskyblue',
                    align='center'),
        cells=dict(values=[list(data_table['age']),
                           list(data_table['sex']),
                           list(data_table['bmi']),
                           list(data_table['children']),
                           list(data_table['smoker']),
                           list(data_table['region']),
                           list(data_table['charges'])],
                   #                line_color='darkslategray',
                   #                fill_color='lightcyan',
                   align=['left', 'center']))
    ])
    fig3.update_layout(title="Top 20 ....",
                       title_x=0.5)
    st.plotly_chart(fig3)


    if st.button("Update Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=0)

        print('Size of x_train = ', X_train.shape)
        print('Size of x_test  = ', X_test.shape)
        print('Size of y_train = ', y_train.shape)
        print('Size of y_test  = ', y_test.shape)

        ## Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # => GradientBoosting Regressor turns out best.
        ### Hyper Parameter Tuning

        param_grid = {
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [1, 2, 5],
            'n_estimators': [1, 5, 10]
        }

        GBR = GradientBoostingRegressor()
        GBR_cv = GridSearchCV(estimator=GBR, param_grid=param_grid, verbose=0)
        GBR_cv.fit(X_train, y_train)

        params = GBR_cv.best_params_

        ### Pipeline
        # from sklearn.pipeline import make_pipeline
        model = GradientBoostingRegressor(learning_rate=params['learning_rate'],
                                          loss=params['loss'],
                                          max_features=params['max_features'],
                                          max_depth=params['max_depth'],
                                          n_estimators=params['n_estimators'])
        # => Fitting Model
        model.fit(X_train, y_train)
        # => Evaluate Model
        evaluate_model(model, X_test, y_test, 'Gradient Boosting Regressor', "Auctual Data")
        # => Save/Pickle Model
        model_name = "model.pk_Charges"
        pickle.dump(model, open(model_name, "wb"))

## Explore
st.sidebar.subheader("Explore")
st.sidebar.write("Check out [PORTFOLIO](https://akanimohod19a.github.io/)")
st.sidebar.write("Check out demo [APPS](https://heartfailurepredictor-afl.herokuapp.com/)")
