# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:24:25 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
# load dataset
data_file=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_FS-NN.nc'
dataset = Measurement()
dataset.from_netcdf(data_file)
print(dataset.summary())

fulldataset = dataset.data

# subclass for split = label_class + deployment_ID
fulldataset['class_label2'] = fulldataset['label_class']+ '__' + fulldataset['deployment_ID']

# groups for splits = label_class + dateHour + deployment_ID
fulldataset['TimeLabel'] = fulldataset['time_min_date'].dt.round("H")
fulldataset['TimeLabel'] = fulldataset['TimeLabel'].apply(lambda x: x.strftime('%Y%m%d%H%M%S'))
fulldataset['group'] = fulldataset['label_class'] + '_' + fulldataset['TimeLabel'] + '_' + fulldataset['deployment_ID'] 

# get dataframe of features
data = fulldataset[dataset.metadata['measurements_name'][0]]
labels = fulldataset['label_class'].to_frame()
groups = fulldataset['group'].to_frame()
sublabels = fulldataset['class_label2'].to_frame()

# ## Transform labels to integers
enc_labels = LabelEncoder()
enc_labels.fit(list(labels['label_class'].values))
labels_ID = enc_labels.transform(list(labels['label_class'].values))
enc_groups = LabelEncoder()
enc_groups.fit(list(groups['group'].values))
groups_ID = enc_groups.transform(list(groups['group'].values))
enc_sublabels = LabelEncoder()
enc_sublabels.fit(list(sublabels['class_label2']))
sublabels_ID = enc_sublabels.transform(list(sublabels['class_label2']))

# # Basic stats on all features
# data_stats = data.describe()
# #print(data_stats)

# # how many NaNs and Infs per column
# data = data.replace([np.inf, -np.inf], np.nan)
# Nnan = data.isna().sum()
# # ax = Nnan.plot(kind='bar',title='Number of NaN/Inf',grid=True)
# # ax.set_ylabel('Number of observations with NaNs/Infs')


# ## clean up ----------------
# # Drop entire columns/features
# data.drop(['uuid'],axis=1, inplace=True) # take out the UUID
# data.drop(['freq_flatness'],axis=1,inplace=True) # nan /inf
# data.drop(['snr'],axis=1,inplace=True) # nan /inf
# # drop observations/rows with NaNs
# data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# # adjust labels accordingly
# labels=labels.iloc[data.index]






# ## display basic stats
# data_stats = data.describe()

# # ## Visualization ----------
# # # box and whisker plots
# # data.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
# # # histograms
# # data.hist()
# # # scatter plot matrix
# # pd.plotting.scatter_matrix(data)

# # # ## Transform labels to integers
# # enc = LabelEncoder()
# # enc.fit(labels)
# # labels = enc.transform(labels)

# ## create group

skf = GroupKFold(n_splits=2)
for train_index, test_index in skf.split(data,labels_ID,groups=groups_ID):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = labels_ID.iloc[train_index], labels_ID.iloc[test_index]

# ## Split-out validation dataset
# X_train, X_validation, Y_train, Y_validation = train_test_split(
#     data, labels,
#     train_size=0.8,
#     test_size=0.2,
#     shuffle=True,
#     stratify=labels,
#     random_state=1)

# ## plot class repartition
# Full_dataset = labels.groupby('label_class')['label_class'].count().to_frame()
# Full_dataset.rename(columns={'label_class':'Full dataset'},inplace=True)
# #Full_dataset.plot.bar(stacked=True, title='Full dataset')
# Train_dataset = Y_train.groupby('label_class')['label_class'].count().to_frame()
# Train_dataset.rename(columns={'label_class':'Training dataset'},inplace=True)
# #Train_dataset.to_frame().plot.bar(stacked=True, title='Train dataset')
# Test_dataset = Y_validation.groupby('label_class')['label_class'].count().to_frame()
# Test_dataset.rename(columns={'label_class':'Evaluation dataset'},inplace=True)
# #Test_dataset.to_frame().plot.bar(stacked=True, title='Validation dataset')
# D=pd.concat([Full_dataset,Train_dataset,Test_dataset],axis=1)
# D.transpose().plot.barh(stacked=True,grid=True)
# plt.tight_layout()
# print(D)

# labels=labels.iloc[data.index]

# ## Build models on train set
# models = []
# models.append(('Dummy', DummyClassifier(strategy="most_frequent")))
# models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
# models.append(('1 Nearest Neighbor', KNeighborsClassifier(n_neighbors=1)))
# models.append(('5 Nearest Neighbor', KNeighborsClassifier(n_neighbors=5)))
# models.append(('10 Nearest Neighbor', KNeighborsClassifier(n_neighbors=10)))
# models.append(('20 Nearest Neighbor', KNeighborsClassifier(n_neighbors=20)))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('Naive Bayes', GaussianNB()))
# models.append(('Random Forest', RandomForestClassifier(n_estimators=10)))
# models.append(('Random Forest', RandomForestClassifier(n_estimators=50)))
# models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))
# models.append(('Random Forest', RandomForestClassifier(n_estimators=500)))
# models.append(('Random Forest', RandomForestClassifier(n_estimators=800)))
# #models.append(('Support Vector Machine', SVC(gamma='auto')))
# #models.append(('Linear Support Vector Machine', LinearSVC(tol=1e-3)))


# # # evaluate each model in turn
# # results = []
# # names = []
# # for name, model in models:
# #  	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# #  	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='f1')
# #  	results.append(cv_results)
# #  	names.append(name)
# #  	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# # Make predictions on validation dataset
# model = LinearDiscriminantAnalysis()
# model.fit(X_train, Y_train)
# predictions_class = model.predict(X_validation)
# predictions_proba = model.predict_proba(X_validation)
# predictions_class_str = enc.inverse_transform(predictions_class)

# # Evaluate predictions
# print(accuracy_score(Y_validation, predictions_class))
# print(confusion_matrix(Y_validation, predictions_class))
# print(classification_report(Y_validation, predictions_class))

# Y_validation2 = np.zeros(len(Y_validation))
# Y_validation2[Y_validation==0]=1
# precision, recall, thresholds = precision_recall_curve(Y_validation2, predictions_proba[:,0])
# pr_auc = metrics.auc(recall, precision)
# #roc_auc = roc_auc_score(Y_validation2, predictions_proba[:,0])
# #print(roc_auc)
# # plot model roc curve
# plt.plot(recall, precision, marker='.', label='Logistic')
# # axis labels
# plt.ylabel('Precision')
# plt.xlabel('Recall')
# plt.title('AUC: '+ "%.2f" % pr_auc)
# # show the legend
# plt.legend()
# # show the plot
# plt.show()



# nfold = 10
# make_balanced = False
# stratification=['label_class','recorder type','deployment_ID']
# groups = hour

# balanced = True

# # > Add group
# # startification

# GroupKFold 
 
