from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

import pygwalker as pyg
from ydata_profiling import ProfileReport

from sqlalchemy.dialects.mssql.information_schema import columns

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer






# fetch dataset from UCI Machine learning Repository Website
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = pd.DataFrame(student_performance.data.features)
y = pd.DataFrame(student_performance.data.targets)
# metadata - explain where dataset from
print(student_performance.metadata)

# variable information - explain information in dataset
print(student_performance.variables)


#restructure train set and test set because of G1 and G2 feature is in target set
X['G1']=y['G1']
X['G2']=y['G2']
y=y['G3']
#print 10 rows in X and y dataset
print(X.head(10))
print(y.head(10))

#Create a student performance profiling report using ydata_profiling
#Combine features and target into 1 dataframe to visualization
student_information = pd.concat([X,y],axis=1)
profile= ProfileReport(student_information,title="Student Performance Profiling Report",minimal=True)
profile.to_file("student_performance_report.html")

#Visualizing dataset by using pygwalker - its outlook is similar like Tableau
walker=pyg.walk(student_information)




#Dropping weak features
drop_cols=['sex','address','famrel','romantic','nursery','internet','activities','Mjob','Fjob']
X=X.drop(columns=drop_cols)





#Print shape of X dataset
print(X.shape)
#Summary missing values
print(X.isnull().sum())

#Splitting dataset into train set, validation set and test set
x_train,x_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=42)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42)


#Group features into 3 type of features, numeric, binary, and nominal
numeric_features=['age','Medu','Fedu','traveltime','studytime','failures','freetime','goout','Dalc','Walc','health','absences','G1','G2']
binary_yes_no_features=['school','famsize','Pstatus','schoolsup','famsup','paid','higher']
nominal_categories_features=['reason','guardian']


def binary_transform(X):
    mapping = {
        'yes': 1, 'no': 0,
        'GP': 1, 'MS': 0,
        'LE3': 1, 'GT3': 0,
        'T': 1, 'A': 0
    }
    # Vectorized replacement
    x_mapped = np.vectorize(mapping.get)(X)
    return x_mapped

yesno_transformer= FunctionTransformer(binary_transform,validate=False)

#using pipeline to combine steps of preprocessing data
#Pipeline for numeric features
numeric_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(missing_values=np.nan,strategy='mean')),
    ('scaler',StandardScaler())
])
#Pipeline for binary features
binary_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('binary',yesno_transformer)
])
#Pipeline for nominal features
nominal_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('nominal',OneHotEncoder())
])



#Using ColumnTransformer to apply transformers to columns of an array or pandas DataFrame
preprocessor= ColumnTransformer(transformers=[
    ('num_feature',numeric_transformer,numeric_features),
    ('bi_feature',binary_transformer,binary_yes_no_features),
    ('nom_feature',nominal_transformer,nominal_categories_features)
])
#Using Pipeline to combine a sequence of predicting model steps
model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('regressor',RandomForestRegressor(random_state=42))
])

#Using train dataset to training model
#Training dataset will automatically fit and transform in pipeline
model.fit(x_train,y_train)
#Predict test set after training model
y_predict=model.predict(x_test)


#Using metrics to evaluate model after training and predicting
print("MAE: {}".format(mean_absolute_error(y_test,y_predict)))
print("MSA: {}".format(mean_squared_error(y_test,y_predict)))
print("R2 score: {}".format(r2_score(y_test,y_predict)))

parameter_grid = [
    {
        'regressor': [RandomForestRegressor(random_state=42)],
        'regressor__n_estimators': [100, 300],
        'regressor__max_depth': [10, 20, None]
    },
    {
        'regressor': [GradientBoostingRegressor(random_state=42)],
        'regressor__n_estimators': [100, 300],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5]
    },
    {
        'regressor': [SVR()],
        'regressor__kernel': ['rbf', 'linear'],
        'regressor__C': [0.1, 1, 10]
    },
    {
        'regressor': [LinearRegression()]
    }
]

#Using hyperparameter tuning
grid_search=GridSearchCV(
    estimator=model,
     param_grid=parameter_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
#Fit model with cross validation method
grid_search.fit(x_train,y_train)

#Print out which model is the best
print("Best Model:", grid_search.best_estimator_)
#Print out which is the best Hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
#Evaluate the best model using R2 score
print("Best CV R² Score:", grid_search.best_score_)

#Using best model to predict validation set
best_model=grid_search.best_estimator_
y_val_predict=best_model.predict(x_val)
#Using metrics to evaluate model after predicting validation set
print("Validation MAE:", mean_absolute_error(y_val, y_val_predict))
print("Validation MSE:", mean_squared_error(y_val, y_val_predict))
print("Validation R2: ", r2_score(y_val,y_val_predict))

#Using test set to evaluate the best model
y_test_pred = best_model.predict(x_test)
#Using metrics to evaluate the best model after predicting test set
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test R2:", r2_score(y_test, y_test_pred))








# import matplotlib.pyplot as plt
#
# feature_names = (
#     numeric_features
#     + binary_yes_no_features
#     + list(best_model.named_steps['preprocessor']
#         .named_transformers_['nom_feature']
#         .named_steps['nominal']
#         .get_feature_names_out(nominal_categories_features))
# )

# importances_feature = best_model.named_steps['regressor'].feature_importances_
#
# sorted_idx = np.argsort(importances_feature)[::-1]
# plt.figure(figsize=(10,6))
# plt.bar(range(len(importances_feature)), importances_feature[sorted_idx])
# plt.xticks(range(len(importances_feature)), np.array(feature_names)[sorted_idx], rotation=90)
# plt.title("Feature Importances from RandomForest")
# plt.show()
