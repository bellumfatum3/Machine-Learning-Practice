# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:36:43 2019

@author: alinaqvim
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import hashlib

housing_path = "C:/Users/alinaqvim/Documents/ML/"
def load_housing_data(housing_path=housing_path):
 csv_path = os.path.join(housing_path, "housing.csv")
 return pd.read_csv(csv_path)
housing = load_housing_data()
#print(housing.describe())
#housing.hist(bins=50, figsize=(20,15))
#plt.show()

#print(housing.head(10))

'''The following code creates an income category
attribute by dividing the median income by 1.5 (to limit the number of income cate‐
gories), and rounding up using ceil (to have discrete categories), and then merging
all the categories greater than 5 into category 5:'''
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

#stratified sampling based on the income category. Sample size is 20%
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]

# measuring stratified sampling bias of original set vs. sampled test_set
# housing is the original set. strat_test_set is the test set (obviously)
# Results are that the strat_test_set is very similarily proportioned to the 
# original from whence it is drawn from  
print(housing["income_cat"].value_counts() / len(housing))
print("--------------")
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# Remove Income Categorical variable from training, test sets
for set in (strat_train_set, strat_test_set):
 set.drop(["income_cat"], axis=1, inplace=True)

# New housing set is a subset of the previous one
# Stratified training set, copied 
housing = strat_train_set.copy()

# Plot longitute and latitude, looking for relationship of population density
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# Same plot, now dividing by 100 to better illustrate and with a colorbar
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population",
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


corr_matrix = housing.corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))


# Plots a scatter matrix of some promising attributes.
# Taken from the corr_matrix 
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# after looking at above graph, we focus on the plot for median_income
# this appears to have best/interesting correlation
housing.plot(kind="scatter", x="median_income", y="median_house_value",
 alpha=0.1)




















