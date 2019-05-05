import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics


dataset = pd.read_csv("data_sets_crime/working_data.csv", encoding='utf-8')
dataset = dataset.drop(dataset.columns[[2]], axis=1)
X = dataset.iloc[:,3:8]
Y = dataset['Ward']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
random_forest_machine = RandomForestClassifier(n_estimators=50)
random_forest_machine.fit(X_train, Y_train)

y_pred = random_forest_machine.predict(X_test)
print(y_pred)

print('r2')
print(metrics.r2_score(Y_test, y_pred))
print('mse')
print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

accuracy_score(Y_test, y_pred)
print('accuracy')
print(accuracy_score(Y_test, y_pred))
print('confusion')
print(confusion_matrix(Y_test, y_pred))
