# Importing the advanced libraries :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Importing the csv data through pandas and read function :
df = pd.read_csv("diabetes.csv")
# This df.head() will import first 5 rows of data.
df.head()
# Telling the number of rows and columns through shape function.
df.shape
# Tells about the information of data through info function.
df.info()
# Tells about type of column through dtypes function.
df.dtypes
# This check whether a column contains null value or not through isnull().sum()
df.isnull().sum()
# This represent bar graph through seaborn library in countplot function.
sns.countplot(x = "Outcome",data = df)
plt.show()
## Showing the number of people who are diabetic and non-diabetic according to their age :
plt.figure(figsize = (20,4))
sns.countplot(x= "Age",hue = "Outcome",data = df)
plt.show()
# Corr() gives the correlation between the target variable and other columns.
df.corr()
# Heatmap function in Seaborn library  gives the graph of correlation between the columns.
sns.heatmap(df.corr(),cbar = True ,square = True , annot = True , fmt = ".1f",annot_kws = {"size":4})
# Importing Machine Learning Libraries:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# drop function will remove the column from particular dataset and stored in X  variable and type of X will be Data Frame.
X = df.drop("Outcome",axis = 1)
type(X)
# Target variable stores in Y variable and type of Y will be series.
Y = df["Outcome"]
type(Y)
# To convert series to frame , to_frame() function is used.
Y = Y.to_frame()
Y
# Model is split into train and test with test size of the model will be 30%.
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)
# RandomForestClassifier() will fit the model.
model = RandomForestClassifier()
model.fit(X_train,Y_train)
# For prediction , predict() function is used which is stored in a variable X_test_prediction.
X_test_prediction = model.predict(X_test)
X_test_prediction
# To check accuracy, accuracy_score() is used which is stored in a variable X_test_accuracy.
X_test_accuracy = accuracy_score(Y_test,X_test_prediction)
# This will gives the Accuracy rate in %.
print("Accuracy :",X_test_accuracy*100,"%")
