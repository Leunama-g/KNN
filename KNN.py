import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = "Dataset(KNN).csv"

# Assign colum names to the dataset
names = ['Number_of_Vehicles','Number_of_Casualties','Day_of_Week','Time','Road_Type','Speed_limit','Junction_Detail','Junction_Control','Pedestrian_Crossing-Human_Control','Pedestrian_Crossing-Physical_Facilities','Light_Conditions','Urban_or_Rural_Area']
_class = ['Accident_Severity']

# Read dataset to pandas dataframe
dataset = pd.read_csv("Dataset(KNN).csv", delimiter=",")

dataset.head()

X = dataset[names].values
y = dataset[_class].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

