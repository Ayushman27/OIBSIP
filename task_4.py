# -*- coding: utf-8 -*-
"""task_4.ipynb


Original file is located at
    https://colab.research.google.com/drive/1rD5ObEI4mooWwc_do5Y_rtT1jJruXTOQ
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/drive/MyDrive/spam.csv', encoding='latin-1')

df.columns

df.info()

df.isnull().sum()

df.isnull().mean()*100

df.drop(columns=df[['Unnamed: 2','Unnamed: 3','Unnamed: 4']],axis=1,inplace=True)
df

df['text length'] = df['v2'].apply(len)

plt.scatter(x=df['text length'],y=df['v1'])
plt.xlabel("Email Type")
plt.ylabel("Text Length")
plt.title("Relationship Between Email Type and Text Length")
plt.show()

sns.barplot(x='v1', y='text length', data=df)

plt.xlabel('Email Type')
plt.ylabel('Text Length')
plt.title('Relationship Between Email Type and Text Length')
plt.show()

sns.boxplot(x='v1', y='text length', data=df)

plt.xlabel('Email Type')
plt.ylabel('Text Length')
plt.title('Relationship Between Email Type and Text Length')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

Vectorizer = CountVectorizer()
x_train_vectorized = Vectorizer.fit_transform(x_train)
x_test_vectorized = Vectorizer.transform(x_test)

classifier = MultinomialNB()
classifier.fit(x_train_vectorized, y_train)

y_pred = classifier.predict(x_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("Classification Report:")
print(class_report)
