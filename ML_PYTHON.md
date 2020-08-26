# WINE_QUALITY-ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
​
d = pd.read_csv(r"C:\Users\DELL\winequality-red.csv")
print(d)
  
d.head()
d.info()

rate = (2,6.5,8)
q_names = ('bad','good')
d['quality'] = pd.cut(d['quality'],bins=rate,labels=q_names)


label_qlt = LabelEncoder()
d['quality'] = label_qlt.fit_transform(d['quality'])
quality
sns.countplot(d['quality'])

d['quality'].value_counts()
d['quality'].value_counts()

X = d.drop('quality',axis=1)
y = d['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

regressor=LogisticRegression()
regressor.fit(X_train,y_train)

score=regressor.score(X_test,y_test)
print('accuracy =' +str(score))

y_pred = regressor.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
    

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
