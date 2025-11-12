# Logistic Regression

# Part 1 - Data Preprocessing

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('data.csv')
dataset.head(10)

# Getting the inputs and output
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X
y

# Creating the Training Set and the Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train
X_test
y_train
y_test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Part 2 - Building and training the model

# Building the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)

# Training the model
model.fit(X_train, y_train)

# Inference

# Making the predictions of the data points in the test set
y_pred = model.predict(sc.transform(X_test))
y_pred

"""
Making the prediction of a single data point with:
1.   Sample Code Number = 1000000
2.   Clump Thickness = 1
3.   Uniformity of Cell Size = 2
4.   Uniformity of Cell Shape = 3
5.   Marginal Adhesion = 4
6.   Single Epithelial Cell Size = 5
7.   Bare Nuclei = 6
8.   Bland Chromatin = 7
9.   Normal Nucleoli = 8
10.  Mitoses = 9
"""

model.predict(sc.transform([[1,2,3,4,5,6,7,8,9]]))

# Part 3: Evaluating the model

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

# Accuracy
(84+47)/(84+47+3+3)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)