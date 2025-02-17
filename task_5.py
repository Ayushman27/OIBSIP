# -*- coding: utf-8 -*-
"""task_5.ipynb


Original file is located at
    https://colab.research.google.com/drive/1uT91f1hQRovJ8NxYG7Smmusx_3thxtw3
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Advertising.csv')

df.head()

df.info()

df.describe()

df

x= df.iloc[:,:-1]
x

y= df.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train

x_test

y_train

y_test

x_train = x_train.astype('int')
x_test = x_test.astype('int')
y_train = y_train.astype('int')
y_test = y_test.astype('int')

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train_scaled,y_train)

y_pred = lr.predict(x_test_scaled)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred, c='red')

plt.angle_spectrum(y_test,y_pred, c='b')
