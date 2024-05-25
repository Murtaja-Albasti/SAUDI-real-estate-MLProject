from colorama import Fore
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#data analyze

House_data = pd.read_csv('./SA_Aqar.csv')
x = House_data.drop(['price'],axis=1)
y = House_data['price']

# train test split the data

x_train_full, x_test_full,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8 , random_state=1)

#preprossecing by pipelines

categorical_columns = [cname for cname in x_train_full.columns if x_train_full[cname].nunique() < 15 and x_train_full[cname].dtypes == 'object']
numerical_columns = [cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64','float64']]

my_cols = categorical_columns + numerical_columns
x_train = x_train_full[my_cols].copy()
x_test = x_test_full[my_cols].copy()

numerical_transformers = SimpleImputer(strategy='constant')

categorical_transformers = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='constant')),
    ('ONH',OneHotEncoder(handle_unknown='ignore'))
])

preproccesor = ColumnTransformer(transformers=[
    ('number',numerical_transformers,numerical_columns),
    ('categorical',categorical_transformers,categorical_columns)
])

# ------------------------------------------

# see for all cities who is more expensive

# plt.figure(figsize=[8,9])
# sns.barplot(data=House_data,y='price',x='city')
# plt.show()


#  - choose the best model by selecting them all or by visualizing the data

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error




# def BestModelAcc(x_train,x_test,y_train,y_test,model):
#     pipline = Pipeline(steps=[
#         ('preprc',preproccesor),
#         ('model',model)
#     ])
#     fit = pipline.fit(x_train,y_train)
#     pred = pipline.predict(x_test)
#     Acc = mean_absolute_error(pred,y_test)
#     print(f'the accurecy of {model} is {Acc/100}')
#
#
# models = [LinearRegression(),DecisionTreeRegressor(random_state=0,criterion='absolute_error'),RandomForestRegressor(random_state=0,n_estimators=500),XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4),SVR(epsilon=0.5)]
#
#
# for model in models :
#     BestModelAcc(x_train,x_test,y_train,y_test,model)

model = RandomForestRegressor(random_state=0,n_estimators=500)

Pipeline = Pipeline(steps=[
    ('prep',preproccesor),
    ('model',model)
])

fit = Pipeline.fit(x_train,y_train)
pred = Pipeline.predict(x_test)

output = pd.DataFrame({'predicted value':pred , 'real value':y_test})
output.to_csv('./Saudi_AQAR',index=False)