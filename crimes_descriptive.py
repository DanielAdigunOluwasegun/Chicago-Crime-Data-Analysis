import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter


df = pd.read_csv("data_sets_crime/working_data.csv", encoding='utf-8')
print(df.shape) 
print(df.dtypes)

print(df.hist(column='FBI Code'))
print(df.hist(column='Community Area'))
print(df.hist(column='Police Districts'))



