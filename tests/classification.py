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
import copy
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
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def add_class_ID(fulldataset, positive_class_label):
    labels = list(set(fulldataset['label_class']))
    # force positive class to be in position 1
    labels.remove(positive_class_label)
    labels.insert(1,positive_class_label)
    # assign class ID (integer with 1 being the positive class)
    IDs = [*range(0,len(labels))]
    fulldataset.insert(0, 'class_ID', -1)
    for n, label in enumerate(labels):
        mask = fulldataset['label_class'] == label
        fulldataset.loc[mask,'class_ID'] = IDs[n]
        class_encoder = pd.DataFrame({'label':labels, 'ID': IDs})
    return fulldataset, class_encoder
    
def add_subclass(fulldataset):
    # subclass for split = label_class + deployment_ID
    fulldataset.insert(0, 'subclass_label', fulldataset['label_class'] + '__' + fulldataset['deployment_ID'])
    labels = list(set(fulldataset['subclass_label']))
    IDs = [*range(0,len(labels))]
    fulldataset.insert(0,'subclass_ID', -1) 
    for n, label in enumerate(labels):
        fulldataset.loc[fulldataset['subclass_label'] == label, 'subclass_ID'] = IDs[n]
    class_encoder = pd.DataFrame({'label':labels, 'ID': IDs})
    return fulldataset, class_encoder

def subclass2class_conversion(fulldataset):
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
    fulldataset.insert(0,'TimeLabel',fulldataset['time_min_date'].dt.round("H").apply(lambda x: x.strftime('%Y%m%d%H%M%S')))
    # subclass for split = label_class + deployment_ID
    fulldataset.insert(0,'group_label',fulldataset['label_class'] + '_' + fulldataset['TimeLabel'] + '_' + fulldataset['deployment_ID'])
    labels = list(set(fulldataset['group_label']))
    IDs = [*range(0,len(labels))]
    fulldataset.insert(0,'group_ID', -1)
    for n, label in enumerate(labels):
        fulldataset.loc[fulldataset['group_label'] == label, 'group_ID'] = IDs[n]
    encoder = pd.DataFrame({'label':labels, 'ID': IDs})
    fulldataset.drop(columns = ['TimeLabel'])
    return fulldataset, encoder

def plot_dataset_distrib(dataset,attr_list=['subclass_label'], title=None):
    title = title + ' (' + str(len(dataset)) + ' data points)'
    nb_plots=len(attr_list)
    fig, ax = plt.subplots(1, nb_plots,
                           sharey=False,
                           constrained_layout=True,)
    for i in range(0,nb_plots):
        if nb_plots == 1:
            current_ax = ax
        else:
            current_ax = ax[i]
        #distrib = data_train.groupby(attr_list[i])[attr_list[i]].count().to_frame()
        distrib = dataset.groupby(attr_list[i])[attr_list[i]].count().to_frame()
        distrib['pct']= distrib[attr_list[i]]/ sum(distrib[attr_list[i]])*100
        #current_ax.bar(distrib.index,distrib['pct'], color='bkrgymc')
        current_ax.bar(distrib.index,distrib['pct'])
        current_ax.set_ylabel('Distribution of data points (%)')
        current_ax.set_title(attr_list[i])
        current_ax.grid()
        current_ax.tick_params(labelrotation=90 )
        #plt.xticks()
    fig.suptitle(title, fontsize=12)

def plot_datasets_groups(data_train, data_test, show=True):
    train_groups = list(set(data_train['group_ID']))
    test_groups = list(set(data_test['group_ID']))
    groups_intersection = list(set(train_groups) & set(test_groups))
    if show:
        plt.figure()
        #plt.bar(['Train set','Test set'],[len(train_groups),len(test_groups)],color='bkrgymc')
        plt.bar(['Train set','Test set'],[len(train_groups),len(test_groups)])
        plt.ylabel('Number of unique groups')
        plt.grid()
        plt.title('Number of shared groups: ' + str(len(groups_intersection)))
    return groups_intersection

def plot_datasets_distrib(data_train, data_test):
    ntrain = len(data_train)
    ntest = len(data_test)
    ntotal = ntrain + ntest
    ntrain = (ntrain/ntotal)*100
    ntest = (ntest/ntotal)*100    
    plt.figure()
    #plt.bar(['Train set','Test set'],[ntrain,ntest], color='bkrgymc')
    plt.bar(['Train set','Test set'],[ntrain,ntest])
    plt.ylabel('% of data points')
    plt.title('Train/test sets data repartition')
    plt.grid()  
    
def main():
    ## define positive class
    positive_class_label ='FS'
    train_ratio = 0.75
    cv_splits = 10
    
    ## LOAD DATSET ---------------------------------------------------------------
    data_file=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_FS-NN.nc'
    dataset = Measurement()
    dataset.from_netcdf(data_file)
    print(dataset.summary())
    
    ## DATA PREPARATION ----------------------------------------------------------
    # features
    features = dataset.metadata['measurements_name'][0] # list of features used for the classification
    #feats = copy.deepcopy(features) # intermediate features list (do not use for classif)
    #feats.append('label_class')
    #feats.append('deployment_ID')
    #feats.append('time_min_date')
    
    # add subclass + IDs
    #data = dataset.data[feats]
    data = dataset.data
    data, _ = add_class_ID(data, positive_class_label)
    data, _ = add_subclass(data)
    subclass2class_table = subclass2class_conversion(data)
    
    # add group ID
    data, group_encoder = add_group(data)
    
    # cleanup
    #data.drop(['deployment_ID'],axis=1,inplace=True)
    #data.drop(['time_min_date'],axis=1,inplace=True)
    #data.drop(['TimeLabel'],axis=1,inplace=True)
    #data.drop(['group_label'],axis=1,inplace=True)
    
    ## DATA CLEAN-UP -------------------------------------------------------------
    # Basic stats on all features
    data_stats = data[features].describe()
    print(data_stats)
    
    # how many NaNs and Infs per column
    data = data.replace([np.inf, -np.inf], np.nan)
    Nnan = data[features].isna().sum()
    ax = Nnan.plot(kind='bar',title='Number of NaN/Inf',grid=True)
    ax.set_ylabel('Number of observations with NaNs/Infs')
    
    # Drop some features with too many NaNs
    features.remove('freq_flatness')
    features.remove('snr')
    features.remove('uuid')
    
    # drop observations/rows with NaNs
    data.dropna(subset=features, axis=0, how='any', thresh=None, inplace=True)
    data_stats2 = data[features].describe()
    
    # ## VISUALIZATION -------------------------------------------------------------
    # # box and whisker plots
    # data[features].plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False)
    # # histograms
    # data[features].hist()
    # # scatter plot matrix
    # pd.plotting.scatter_matrix(data[features])
    
    ## SPLIT DATA INTO TRAIN & TEST SETS ------------------------------------------
    n_splits = round(1/(1-train_ratio))
    skf = StratifiedGroupKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(data, data['subclass_ID'],groups=data['group_ID']):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        break
    
    # plot class repartition
    plot_datasets_distrib(data_train, data_test)
    plot_dataset_distrib(data,attr_list=['subclass_label','label_class'],title='Full dataset')
    plot_dataset_distrib(data_train,attr_list=['subclass_label','label_class'],title='Training set')
    plot_dataset_distrib(data_test,attr_list=['subclass_label','label_class'],title='Test set')
    
    # verify groups are not used in both datasets
    groups_intersection = plot_datasets_groups(data_train, data_test, show=True)
    
    ## DEFINITION OF CLASSIFIERS -------------------------------------------------
    models = []
    models.append(('Dummy', DummyClassifier(strategy="most_frequent")))
    models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
    
    ## CROS VALIDATION ON TRAIN SET ----------------------------------------------
    cv_results = pd.DataFrame({'CV_iter':[],'classifier':[],'uuid':[],'Y_true':[],'Y_pred':[],'Y_prob':[]})
    skf = StratifiedGroupKFold(n_splits=cv_splits)    
    it=-1
    for cv_train_index, cv_val_index in skf.split(data_train, data_train['subclass_ID'],groups=data_train['group_ID']):
        it+=1
        # Split data train vs validation
        cv_data_train, cv_data_val = data_train.iloc[cv_train_index], data_train.iloc[cv_val_index]
        groups_intersection = plot_datasets_groups(cv_data_train, cv_data_val, show=False)        
        # CV summary counts
        distrib_train = cv_data_train.groupby('label_class')['label_class'].count().to_frame()
        distrib_train.rename(columns={'label_class':'train'}, inplace=True)
        distrib_val = cv_data_val.groupby('label_class')['label_class'].count().to_frame()
        distrib_val.rename(columns={'label_class':'Validation'}, inplace=True)
        cv_summary = pd.concat([distrib_train, distrib_val],axis=1)
        # display CV info
        print(' ')
        print(' ')
        print('Cross validation #', str(it) + ' ---------------------------------------')
        print(cv_summary)
        print(' ')
        print('Intersecting groups:' + str(len(groups_intersection)))

        #plot_dataset_distrib(cv_data_train,attr_list=['subclass_label','label_class'],title='Training set (CV #' + str(it) +')' )
        #plot_dataset_distrib(cv_data_val,attr_list=['subclass_label','label_class'],title='Evaluation set (CV #' + str(it) +')' )
        # reformat data
        X_train = cv_data_train[features] # features
        Y_train = cv_data_train['class_ID'] #labels
        X_val = cv_data_val[features] # features
        Y_val = cv_data_val['class_ID'] #labels
        Y_uuid = cv_data_val['uuid']
        
        # feature normalization
        print('Classifiers:')
        for model_name, model in models:
            print('-> ' + model_name)
            # train model
            model.fit(X_train, Y_train)
            # predict
            pred_class = model.predict(X_val)
            pred_prob = model.predict_proba(X_val)
            # stack info
            tmp = pd.DataFrame({'CV_iter':[],'classifier':[],'uuid':[],'Y_true':[],'Y_pred':[],'Y_prob':[]})
            tmp['uuid']= cv_data_val['uuid']
            tmp['CV_iter'] = it
            tmp['classifier'] = model_name
            tmp['Y_true'] = Y_val
            tmp['Y_pred'] = pred_class
            tmp['Y_prob'] = pred_prob[:,1]
            cv_results = pd.concat([cv_results,tmp],ignore_index=True)
            

    # evaluate predictions
    print('stop')
    precision, recall, thresholds = precision_recall_curve(Y_val, pred_prob[:,0])
    pr_auc = metrics.auc(recall, precision)
    f1 = f1_score(Y_val, pred_class, average='binary')    
    CR = classification_report(Y_val, pred_class)
    CM = confusion_matrix(Y_val, pred_class)
    

if __name__ == "__main__":
    main()
    
    
    
    
# print(accuracy_score(Y_validation, predictions_class))
# print(confusion_matrix(Y_validation, predictions_class))
# print(classification_report(Y_validation, predictions_class))
    
# summary = data_train.pivot_table(index='',
#                                  columns=columns,
#                                  aggfunc='size',
#                                  fill_value=0)
#         # Add a "Total" row and column
#         summary.loc['Total']= summary.sum()
#         summary['Total']= summary.sum(axis=1)
        


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



