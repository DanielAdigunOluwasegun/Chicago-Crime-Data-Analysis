import pandas as pd 
import os
import pandas as pd 
import csv
import numpy as np 

if not os.path.exists("data_sets_crime"):
	os.mkdir("data_sets_crime")
datasets = pd.read_csv("Crimes_-_2001_to_present.csv", encoding='utf-8')
# print(dataset)

dataset = datasets[69414:335917]
dataset.to_csv("data_sets_crime/raw_data1.csv", index=False, encoding='utf-8')
# 2017-2018

df = pd.read_csv("data_sets_crime/raw_data1.csv", encoding='utf-8')
df = df.drop(df.columns[[0, 1, 3, 4, 5, 6, 7, 8, 9, 15, 16, 18, 19, 20, 22, 24, 26, 27, 29]], axis=1)
df.to_csv("data_sets_crime/raw_data2.csv", index=False, encoding='utf-8')


df = pd.read_csv("data_sets_crime/raw_data2.csv", encoding='utf-8')
df['FBI Code'] = pd.to_numeric(df['FBI Code'], errors='coerce')
df.dropna(inplace=True)
print(df.info())
df.to_csv("data_sets_crime/working_data.csv", index=False, encoding='utf-8')


