import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv("diabetes.csv")
df
df.shape
df.info()
df.dtypes
df.isnull().sum()
df.columns
sns.countplot(x = "Outcome",data = df)
plt.show()
## Showing the number of people who are diabetic and non-diabetic according to their age :
plt.figure(figsize = (20,4))
sns.countplot(x= "Age",hue = "Outcome",data = df)
plt.show()
df.corr()
sns.heatmap(df.corr(),cbar = True ,square = True , annot = True , fmt = ".1f",annot_kws = {"size":4})
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X = df.drop("Outcome",axis = 1)
type(X)
Y = df["Outcome"]
type(Y)
Y = Y.to_frame()
Y
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)
model = RandomForestClassifier()
model.fit(X_train,Y_train)
X_test_prediction = model.predict(X_test)
X_test_prediction
X_test_accuracy = accuracy_score(Y_test,X_test_prediction)
print("Accuracy :",X_test_accuracy*100,"%")