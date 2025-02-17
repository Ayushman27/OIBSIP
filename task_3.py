# -*- coding: utf-8 -*-
"""task_3.ipynb


Original file is located at
    https://colab.research.google.com/drive/1AINkKyGJkQndAZ766U8xI586dzPhnEZk
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('/content/drive/MyDrive/car data.csv')

df.head()

df.info()

df.describe()

inputs = df.drop(['Car_Name','Selling_Price'],axis='columns')
target = df['Selling_Price']
inputs

from sklearn.preprocessing import LabelEncoder
Numerics = LabelEncoder()

inputs['fuel_n'] = Numerics.fit_transform(inputs['Fuel_Type'])
inputs['Transmission'] = Numerics.fit_transform(inputs['Transmission'])
inputs

inputs_n = inputs.drop(['Fuel_Type','Transmission'],axis='columns')
inputs_n

from sklearn.preprocessing import LabelEncoder
Numerics = LabelEncoder()

for column in inputs_n.select_dtypes(include=['object']).columns:
    inputs_n[column] = Numerics.fit_transform(inputs_n[column])

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(inputs_n,target)

pred = model.predict([[2013,430000,1,1,0,1]])
print(pred)

df

training_data_predication = model.predict(inputs_n)

from sklearn import metrics
error_score = metrics.r2_score(target, training_data_predication)
print("R squared Error : ", error_score)

plt.scatter(target, training_data_predication)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
