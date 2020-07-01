# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:24:25 2020

@author: xavier.mouy
"""
from ecosound.core.measurement import Measurement
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
data_file=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_FS-NN.nc'
dataset = Measurement()
dataset.from_netcdf(data_file)
print(dataset.summary())

# get dataframe of features
data = dataset.data[dataset.metadata['measurements_name'][0]]
labels = dataset.data['label_class']

# Basic stats on all features
data_stats = data.describe()
print(data_stats)

# clean up ----------------
data.drop(['uuid'],axis=1, inplace=True) # take out the UUID
data.drop(['time_flatness'],axis=1,inplace=True) # nan /inf
data.drop(['freq_flatness'],axis=1,inplace=True) # nan /inf
data_stats = data.describe()

# ## Visualization ----------
# # box and whisker plots
# data.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
# # histograms
# data.hist()
# # scatter plot matrix
# pd.plotting.scatter_matrix(data)

# Split-out validation dataset
array = data.values
X = data.values
y = labels.values
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X,y,
    train_size=0.8,
    test_size=0.2,
    shuffle = True,
    stratify = ,
    random_state=1)

# ## summary
# print(meas.data.describe())

# ## visualization
# #Box plot
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# # histogram
# meas.data.hist()
# # bivariable scatter plot
# pd.plotting.scatter_matrix(meas.data)



# nfold = 10
# make_balanced = False
# stratification=['label_class','recorder type','deployment_ID']
# groups = hour

# balanced = True

# # > Add group
# # startification

# GroupKFold 
 

# # Split-out validation dataset
# array = dataset.values
# X = array[:,0:4]
# y = array[:,4]
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# 	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
# 	results.append(cv_results)
# 	names.append(name)
# 	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))