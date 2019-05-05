# Econ_9000finals
Finals Chicago crime data
The raw data was gotten from https://we.tl/t-9q2e1SRVpR), this dataset represent crime data for the city of Chicago from 2001-to date(with the exemption of sensitive information especially for active cases).

the following steps where carried out using the code in crimes_data_cleaning.py
first step was to reduce the data to cover my period of interest 2017-2018, which we read into a directory created for all our datasets(data_sets_scrime)
then we convert the categorical columns (Arrest and Domestic) into numerics 0 = false and 1= true, this eneables us to have a uniform dataset that can be run on any of the models used for the project, the finalworking file is named working_data.csv

Next we run regression analysis using the file crimes_linearreg.py
followed by the KNN model using the file crimes_knn.py
then Random forest model using crimes_random_forest.py
and finally the decision tree model using the file crimes_decisiontree.py

tuning was done for the KNN and Random forest model to enable us use the best parameters in our analysis. 

Also attached is a pdf file that contains the description of the data set and assumptiuons made when extracting and making analysis.
