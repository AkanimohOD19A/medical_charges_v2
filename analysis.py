import numpy as np
import pandas as pd
import gspread as gs
import pickle
from oauth2client.service_account import ServiceAccountCredentials
import warnings

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
scope = ['https://www.googleapis.com/auth/spreadsheets',
         'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials. \
    from_json_keyfile_name("gs_credentials.json", scope)
client = gs.authorize(credentials)

# Get the instance of the Spreadsheet
sheet = client.open('INSURANCE')

# Get the first sheet of the Spreadsheet - where we have the data
sheet_instance = sheet.get_worksheet(0)
records_data = sheet_instance.get_all_records()  ## json

# Convert the json to dataframe
df = pd.DataFrame.from_dict(records_data)
## Rename columns
df.columns = ['age', 'sex', 'bmi',
              'children', 'smoker',
              'region', 'charges', 'entry_date', 'user_id']

df.drop(columns=['entry_date', 'user_id'], inplace=True)

# Engineering
## Feature Engineering
from sklearn.preprocessing import LabelEncoder

sex_map = {'male': 1, 'female': 0}
df['sex'] = df['sex'].map(sex_map).astype('int64')

smoker_map = {'yes': 1, 'no': 0}
df['smoker'] = df['smoker'].map(smoker_map).astype('int64')

LE = LabelEncoder()
df['region'] = LE.fit_transform(df['region'])

## Train-Test Split
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

X = df.drop(columns='charges').values
y = df['charges'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

print('Size of x_train = ', X_train.shape)
print('Size of x_test  = ', X_test.shape)
print('Size of y_train = ', y_train.shape)
print('Size of y_test  = ', y_test.shape)

## Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# => GradientBoosting Regressor turns out best.
### Hyper Parameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'loss': ['ls', 'lad', 'huber', 'quantile'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [1, 2, 10],
    'n_estimators': [10, 15, 20]
}

GBR = GradientBoostingRegressor()
GBR_cv = GridSearchCV(estimator=GBR, param_grid=param_grid, verbose=2)
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
