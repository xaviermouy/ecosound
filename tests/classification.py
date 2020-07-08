# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:24:25 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
from ecosound.classification.CrossValidation import StratifiedGroupKFold
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


def add_class_ID(fulldataset):
    labels = list(set(fulldataset['label_class']))
    IDs = [*range(0,len(labels))]
    fulldataset['class_ID']=-1
    for n, label in enumerate(labels):
        fulldataset.loc[fulldataset['label_class'] == label, 'class_ID'] = IDs[n]
    class_encoder = pd.DataFrame({'label':labels, 'ID': IDs})
    return fulldataset, class_encoder
    
def add_subclass(fulldataset):
    # subclass for split = label_class + deployment_ID
    fulldataset['subclass_label'] = fulldataset['label_class'] + '__' + fulldataset['deployment_ID']
    labels = list(set(fulldataset['subclass_label']))
    IDs = [*range(0,len(labels))]
    fulldataset['subclass_ID']=-1
    for n, label in enumerate(labels):
        fulldataset.loc[fulldataset['subclass_label'] == label, 'subclass_ID'] = IDs[n]
    class_encoder = pd.DataFrame({'label':labels, 'ID': IDs})
    return fulldataset, class_encoder

def subclass2class_converion(fulldataset):
    subclass_labels = list(set(fulldataset['subclass_label']))
    subclass_IDs = [*range(0,len(subclass_labels))]
    class_labels = []
    class_IDs = []
    for n, subclass_label in enumerate(subclass_labels):
        idx = fulldataset.index[fulldataset['subclass_label'] == subclass_labels[n]].tolist()
        class_labels.append(fulldataset.iloc[idx[0]]['label_class'])
        class_IDs.append(fulldataset.iloc[idx[0]]['class_ID'])      
    class_encoder = pd.DataFrame({'subclass_labels': subclass_labels, 'subclass_ID': subclass_IDs, 'class_labels': class_labels, 'class_IDs': class_IDs})
    return class_encoder

def add_group(fulldataset):
    # # groups for splits = label_class + dateHour + deployment_ID
    fulldataset['TimeLabel'] = fulldataset['time_min_date'].dt.round("H").apply(lambda x: x.strftime('%Y%m%d%H%M%S'))
    # subclass for split = label_class + deployment_ID
    fulldataset['group_label'] = fulldataset['label_class'] + '_' + fulldataset['TimeLabel'] + '_' + fulldataset['deployment_ID'] 
    labels = list(set(fulldataset['group_label']))
    IDs = [*range(0,len(labels))]
    fulldataset['group_ID']=-1
    for n, label in enumerate(labels):
        fulldataset.loc[fulldataset['group_label'] == label, 'group_ID'] = IDs[n]
    encoder = pd.DataFrame({'label':labels, 'ID': IDs})
    fulldataset.drop(columns = ['TimeLabel'])
    return fulldataset, encoder
    

# load dataset

data_file=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_FS-NN.nc'
#data_file=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset.nc'
dataset = Measurement()
dataset.from_netcdf(data_file)
print(dataset.summary())

## DATA PREPARATION ----------------------------------------------------------
# features
feats = dataset.metadata['measurements_name'][0]
feats.append('label_class')
feats.append('deployment_ID')
feats.append('time_min_date')

# add subclass + IDs
data = dataset.data[feats]
data, _ = add_class_ID(data)
data, _ = add_subclass(data)
subclass2class_table = subclass2class_converion(data)

# add group ID
data, group_encoder = add_group(data)

# cleanup
data.drop(['uuid'],axis=1,inplace=True)
data.drop(['class_ID'],axis=1,inplace=True)
data.drop(['label_class'],axis=1,inplace=True)
data.drop(['subclass_label'],axis=1,inplace=True)
data.drop(['deployment_ID'],axis=1,inplace=True)
data.drop(['time_min_date'],axis=1,inplace=True)
data.drop(['TimeLabel'],axis=1,inplace=True)
data.drop(['group_label'],axis=1,inplace=True)

## DATA CLEAN-UP -------------------------------------------------------------

# Basic stats on all features
data_stats = data.describe()
#print(data_stats)

# how many NaNs and Infs per column
data = data.replace([np.inf, -np.inf], np.nan)
Nnan = data.isna().sum()
ax = Nnan.plot(kind='bar',title='Number of NaN/Inf',grid=True)
ax.set_ylabel('Number of observations with NaNs/Infs')

# Drop some fetaures with too many NaNs
data.drop(['freq_flatness'],axis=1,inplace=True) # nan /inf
data.drop(['snr'],axis=1,inplace=True) # nan /inf
# drop observations/rows with NaNs
data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

data_stats2 = data.describe()

# ## VISUALIZATION -------------------------------------------------------------
# # box and whisker plots
# data.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
# # histograms
# data.hist()
# # scatter plot matrix
# pd.plotting.scatter_matrix(data)

## SPLIT DATA INTO TRAIN & TEST SETS ------------------------------------------

train_ratio = 0.75
n_splits = round(1/(1-train_ratio))
skf = StratifiedGroupKFold(n_splits=n_splits)
for train_index, test_index in skf.split(data, data['subclass_ID'],groups=data['group_ID']):
    #print("TRAIN:", train_index, "TEST:", test_index)   
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    break

print('yoo')

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
 
