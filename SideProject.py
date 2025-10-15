from ucimlrepo import fetch_ucirepo
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import pygwalker as pyg
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score








# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = pd.DataFrame(student_performance.data.features)
y = pd.DataFrame(student_performance.data.targets)

#create a student performance profiling report
student_information = pd.concat([X,y],axis=1)
# print(student_information.head(10))
# profile= ProfileReport(student_information,title="Student Performance Profiling Report",minimal=True)
# profile.to_file("student_performance_report.html")

#visualizing dataset by using pygwalker
# walker=pyg.walk(student_information)



#checking correlation if weak or not nursery,activities, internet
#Dropping weak features
# drop_cols=['sex','school','address','reason','famrel','romantic','nursery','internet','activities','Mjob','Fjob']
# X=X.drop(columns=drop_cols)




# metadata
# print(student_performance.metadata)

# variable information
# print(student_performance.variables)
print(y.head(10))
# print(X.head(10))
print(X.shape)
print(X.isnull().sum())

#splitting dataset into train set, validation set and test set
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.5,random_state=42)


#group features into 3 type of features, numeric, binary, and nominal ,'G1','G2'
numeric_features=['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']
binary_yes_no_features=['sex','school','address','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
nominal_categories_features=['Mjob','Fjob','reason','guardian']

yesno_transformer= FunctionTransformer(lambda x: (x=='yes').astype(int),validate=False)
preprocessor= ColumnTransformer([
    ('numeric',StandardScaler(),numeric_features),
    ('binary',yesno_transformer,binary_yes_no_features),
    ('nominal',OneHotEncoder(),nominal_categories_features)
])

preprocessor.fit(x_train)
x_train_preprocessed=preprocessor.transform(x_train)
x_test_preprocessed=preprocessor.transform(x_test)

rf=RandomForestRegressor()
rf.fit(x_train_preprocessed,y_train)
y_pred=rf.predict(x_test_preprocessed)
print("R2 on test set: ", r2_score(y_test,y_pred))





