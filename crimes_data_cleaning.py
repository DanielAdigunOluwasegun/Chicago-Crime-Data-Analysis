import pandas as pd 
import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder



if not os.path.exists("data_sets_crime"):
	os.mkdir("data_sets_crime")
datasets = pd.read_csv("Crimes_-_2001_to_present.csv", encoding='utf-8')
# print(dataset)

dataset = datasets[69414:335917]
dataset.to_csv("data_sets_crime/raw_data1.csv", index=False, encoding='utf-8')
# 2017-2018

df = pd.read_csv("data_sets_crime/raw_data1.csv", encoding='utf-8')
df = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29]], axis=1)
df.to_csv("data_sets_crime/raw_data2.csv", index=False, encoding='utf-8')


df = pd.read_csv("data_sets_crime/raw_data2.csv", encoding='utf-8')
df['FBI Code'] = pd.to_numeric(df['FBI Code'], errors='coerce')
df.dropna(inplace=True)
categorical_feature_mask = df.dtypes==bool
categorical_cols = df.columns[categorical_feature_mask].tolist()
le = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(lambda col:
	le.fit_transform(col))
df[categorical_cols].head(10)
print(df.info())
df.to_csv("data_sets_crime/working_data.csv", index=False, encoding='utf-8')


