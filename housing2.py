# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:39:10 2019

@author: pwnrz
"""

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

housing_path = "C:/Users/pwnrz/Documents/ML/"
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


# plot correlation matrix of the stratified training set
corr_matrix = housing.corr()
(corr_matrix["median_house_value"].sort_values(ascending=False))


# Scatter matrix for non-linear relationships
from pandas.tools.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# Scatter matrix focusing on subplot of median_income
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)

# Create 3 new attributes and plot all 3 against income again. 
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
print("reee")
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


## Data cleaning. Don't need to explain TBH. from pg 60
# rever to clean data set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing = housing.drop("total_bedrooms", axis=1) # option 2


# Create ocean proximity variable for how close stuff is to ocean from 
# the text variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded

# One hot encoding method. pg 63
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot.toarray())  # converts to array instead of storing as a 
 #scipy matrix
 
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
 ('imputer', Imputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

# Dataframe selector doesn't work. use columntransform instead.
# located here:https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

from sklearn.base import BaseEstimator, TransformerMixin
# Create a class to select numerical or categorical columns 
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy="median")

##Training and Evaluating on the Training Set

# Linear regression model, simple
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)