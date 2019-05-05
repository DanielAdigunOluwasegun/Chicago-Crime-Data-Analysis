import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np

dataset = pd.read_csv("data_sets_crime/working_data.csv", encoding='utf-8')
print(dataset.head())
dataset = dataset.drop(dataset.columns[[2]], axis=1)
X = dataset.iloc[:,3:8]
Y = dataset['Ward']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

print('knn_neighbors5')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print('r2')
print(metrics.r2_score(Y_test, y_pred))
print('mse')
print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
accuracy_score(Y_test, y_pred)
print('accuracy5')
print(accuracy_score(Y_test, y_pred))
print('confusion5')
print(confusion_matrix(Y_test, y_pred))

print('knn_neighbors6')
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print('r2')
print(metrics.r2_score(Y_test, y_pred))
print('mse')
print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
accuracy_score(Y_test, y_pred)
print('accuracy6')
print(accuracy_score(Y_test, y_pred))
print('confusion6')
print(confusion_matrix(Y_test, y_pred))

print('knn_neighbors7')
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print('r2')
print(metrics.r2_score(Y_test, y_pred))
print('mse')
print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
accuracy_score(Y_test, y_pred)
print('accuracy7')
print(accuracy_score(Y_test, y_pred))
print('confusion7')
print(confusion_matrix(Y_test, y_pred))


print('knn_neighbors8')
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print('r2')
print(metrics.r2_score(Y_test, y_pred))
print('mse')
print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
accuracy_score(Y_test, y_pred)
print('accuracy8')
print(accuracy_score(Y_test, y_pred))
print('confusion8')
print(confusion_matrix(Y_test, y_pred))