from sklearn.linear_model import LinearRegression
from sklearn. metrics import mean_squared_error
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt
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

plt.scatter(Y_test, y_pred)
plt.xlabel('Y_test')
plt.ylabel('y_pred')

plt.savefig('scatterplot.png')

metrics.r2_score(Y_test, y_pred)
print(metrics.r2_score(Y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

