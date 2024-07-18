import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import tree
from sklearn.tree import plot_tree

car_data = pd.read_csv('car_data.csv')
car_data.head()
car_data.tail()

print ("Dataset Lenght:: ", len(car_data))

print("Dataset Shape: ", car_data.shape)

car_data['Gender'] = car_data['Gender'].map({'Male': 0, 'Female': 1})
car_data = car_data.drop('User ID', axis=1)

car_data

car_data.isnull().sum()

X = car_data.values[:, 0:3]
Y = car_data.values[:, 3]

type(X)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
        accuracy_score(y_test, y_pred)*100)
    print("Report : ",
        classification_report(y_test, y_pred))
    
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred_en = clf_entropy.predict(X_test)
y_pred_en

cal_accuracy(y_test,y_pred_en)

clf_entropy
plt.figure(figsize=(50,25))
plot_tree(clf_entropy, 
        filled=True, 
        rounded=True, 
        class_names=["0", "1"],
        feature_names=["Gender", "Age", "AnnualSalary"])
plt.show()

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
# Performing training
clf_gini.fit(X_train, y_train)

y_pred_gi = clf_gini.predict(X_test)
y_pred_gi

cal_accuracy(y_test,y_pred_gi)

clf_gini
plt.figure(figsize=(50,25))
plot_tree(clf_gini, 
        filled=True, 
        rounded=True, 
        class_names=["0", "1"],
        feature_names=["Gender", "Age", "AnnualSalary"])
plt.show()