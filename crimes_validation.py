import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt 
from sklearn.model_selection import KFold 
from sklearn.linear_model import LinearRegression 
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn import metrics


dataset = pd.read_csv("data_sets_crime/working_data.csv", encoding='utf-8')
print(dataset.head())
dataset = dataset.drop(dataset.columns[[2]], axis=1)
X = dataset.drop('Ward', axis=1)
Y = dataset['Ward']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

linreg = linear_model.LinearRegression()
linreg.fit(X_train, Y_train)
print(linreg.coef_)

y_pred = linreg.predict(X_test)
print(y_pred)

metrics.r2_score(Y_test, y_pred)
print(metrics.r2_score(Y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

# lookup  sklearn documentation


data = dataset.values
print(data)

kfold_machine = KFold(n_splits = 4)
kfold_machine.get_n_splits(data)
print(kfold_machine)

for training_index, test_index in kfold_machine.split(data):
	print("Training: ",training_index)
	print("Test: ", test_index)
	X_train, X_test = X[training_index], X[test_index]
	Y_train,Y_test = Y[training_index], Y[test_index]
	linear_machine = linear_model.LinearRegression()
	linear_machine.fit(X_train,Y_train)
	predict = linear_machine.predict(X_test)
	print(metrics.r2_score(Y_test, predict))