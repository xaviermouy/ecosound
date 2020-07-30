# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:24:25 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
from ecosound.classification.CrossValidation import StratifiedGroupKFold
from ecosound.classification.CrossValidation import RepeatedStratifiedGroupKFold

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import pickle
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

def calc_tp_fp_fn_tn(Y_true,Y_prob,threshold):
    
    # init
    tp = np.zeros(len(Y_prob))
    fp = np.zeros(len(Y_prob))
    fn = np.zeros(len(Y_prob))
    tn = np.zeros(len(Y_prob))
    # thresholding
    Y_pred = np.zeros(len(Y_prob))
    Y_pred[Y_prob>=threshold] = 1
    idx=-1
    for true, pred in zip(Y_true, Y_pred):
        idx+=1
        if (true == 1) & (pred == 1): # true positive
            tp[idx]=1
        elif (true == 0) & (pred == 1): # false positive
            fp[idx]=1
        elif (true == 1) & (pred == 0): # false negative
            fn[idx]=1
        elif (true == 0) & (pred == 0): # true negative
            tn[idx]=1
    return tp, fp, fn, tn

def calc_performance_metrics(Y_true,Y_prob,thresholds=np.arange(0,1.1,0.1)):
    n = len(thresholds)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1_score = np.zeros(n)
    AUC_PR=0
    AUC_f1=0
    for idx, threshold in enumerate(thresholds):
        tp, fp, fn, tn = calc_tp_fp_fn_tn(Y_true,Y_prob,threshold=threshold)
        tp_tot = sum(tp)
        fp_tot = sum(fp)
        fn_tot = sum(fn)
        if (tp_tot == 0) | (fp_tot == 0):
            precision[idx] = np.nan
        else:
            precision[idx] = tp_tot /(tp_tot + fp_tot)
        recall[idx] = tp_tot /(tp_tot + fn_tot)
        f1_score[idx] = (2*precision[idx]*recall[idx]) / (precision[idx]+recall[idx])
    
    AUC_PR = metrics.auc(recall, precision)
    AUC_f1 = metrics.auc(thresholds, f1_score)
    out = pd.DataFrame({'thresholds': thresholds,'precision':precision,'recall':recall,'f1-score':f1_score})
    #out['AUC-PR'] = AUC_PR
    #out['AUC-f1'] = AUC_f1
    return out

def cross_validation(data_train, models, features, cv_splits=10,cv_repeats=10):
    cv_predictions = pd.DataFrame({'CV_iter':[],'classifier':[],'uuid':[],'Y_true':[],'Y_pred':[],'Y_prob':[]})
    cv_performance = pd.DataFrame({'CV_iter':[],'classifier':[],'precision':[],'recall':[],'f1-score':[],'thresholds':[]})
    skf = RepeatedStratifiedGroupKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=1)
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
        Norm_mean = X_train.mean()
        Norn_std = X_train.std()
        X_train = (X_train-Norm_mean)/Norn_std
        X_val = (X_val-Norm_mean)/Norn_std
        
        # Train and predict
        print('Classifiers:')
        for model_name, model in models:
            print('-> ' + model_name)
            # train model
            model.fit(X_train, Y_train)
            # predict
            pred_class = model.predict(X_val)
            pred_prob = model.predict_proba(X_val)
            # stack prediction info
            tmp = pd.DataFrame({'CV_iter':[],'classifier':[],'uuid':[],'Y_true':[],'Y_pred':[],'Y_prob':[]})
            tmp['uuid']= cv_data_val['uuid']
            tmp['CV_iter'] = it
            tmp['classifier'] = model_name
            tmp['Y_true'] = Y_val
            tmp['Y_pred'] = pred_class
            tmp['Y_prob'] = pred_prob[:,1]
            cv_predictions = pd.concat([cv_predictions,tmp],ignore_index=True)
            # calculate performance metrics            
            performance = calc_performance_metrics(Y_val.values,pred_prob[:,1])
            performance['classifier'] = model_name
            performance['CV_iter'] = it
            cv_performance = pd.concat([cv_performance,performance],ignore_index=True)
    return cv_predictions, cv_performance

def summarize_performance(cv_performance, threshold=0.5):
    # evaluate predictions
    summary = pd.DataFrame({'Classifier':[],'Precision (mean)':[],'Precision (std)':[],'Recall (mean)':[],'Recall (std)':[],'f1-score (mean)':[],'f1-score (std)':[]})
    # plot PR curves 
    classifiers = list(set(cv_performance['classifier']))
    cv_iterations = list(set(cv_performance['CV_iter']))
    for classifier in classifiers:
        temp_classif = cv_performance[cv_performance['classifier']==classifier]
        temp_classif = temp_classif[temp_classif['thresholds']==threshold]
        p_mean = round(temp_classif['precision'].mean(),3)
        p_std = round(temp_classif['precision'].std(),3)
        r_mean = round(temp_classif['recall'].mean(),3)
        r_std = round(temp_classif['recall'].std(),3)
        f_mean = round(temp_classif['f1-score'].mean(),3)
        f_std = round(temp_classif['f1-score'].std(),3)
        tmp = pd.DataFrame({'Classifier': [classifier],'Precision (mean)': [p_mean],'Precision (std)':[p_std],'Recall (mean)':[r_mean],'Recall (std)':[r_std],'f1-score (mean)':[f_mean],'f1-score (std)':[f_std]})
        summary = pd.concat([summary, tmp], ignore_index=True)
    return summary.T

def plot_PR_curves(cv_performance):
    # plot PR curves 
    classifiers = list(set(cv_performance['classifier']))
    fig, ax = plt.subplots(1, 1,
                               sharey=False,
                               constrained_layout=True,)
    for classifier in classifiers:
        temp = cv_performance[cv_performance['classifier']==classifier]
        temp2 = temp.groupby(['thresholds']).mean()
        ax.plot(temp2['recall'],temp2['precision'], label=classifier)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_title('Average Precision and Recall curve')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.legend()

def plot_F_curves(cv_performance):
    # plot PR curves 
    classifiers = list(set(cv_performance['classifier']))
    fig, ax = plt.subplots(1, 1,
                               sharey=False,
                               constrained_layout=True,)
    for classifier in classifiers:
        temp = cv_performance[cv_performance['classifier']==classifier]
        temp2 = temp.groupby(['thresholds']).mean()
        ax.plot(temp2.index,temp2['f1-score'], label=classifier)
    ax.set_ylabel('f1-score')
    ax.set_xlabel('Decision threshold')
    ax.set_title('Average f1-score curve')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.legend()

def classification_train(X_train, Y_train, model):
    model_trained = model.fit(X_train, Y_train)
    return model_trained

def classification_predict(X_test, model_trained):
    pred_class = model_trained.predict(X_test)
    pred_prob = model_trained.predict_proba(X_test)
    return pred_class, pred_prob[:,1]    
    
def main():
    ## define positive class
    positive_class_label ='FS'
    model_filename = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Classification\RF300_model.sav'
    train_ratio = 0.75
    cv_splits = 5#10
    cv_repeats = 1
    
    ## LOAD DATSET ---------------------------------------------------------------
    data_file=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_FS-NN.nc'
    dataset = Measurement()
    dataset.from_netcdf(data_file)
    print(dataset.summary())

    ## DATA PREPARATION ----------------------------------------------------------
    # features
    features = dataset.metadata['measurements_name'][0] # list of features used for the classification
    # data
    data = dataset.data
    # drop FS observations at Mill Bay 
    indexNames = data[(data['label_class'] == 'FS') & (data['location_name'] == 'Mill bay') ].index
    data.drop(indexNames , inplace=True)
    # add subclass + IDs
    data, class_encoder = add_class_ID(data, positive_class_label)
    data, _ = add_subclass(data)
    #subclass2class_table = subclass2class_conversion(data)
    # add group ID
    data, group_encoder = add_group(data)

    ## DATA CLEAN-UP -------------------------------------------------------------
    # Basic stats on all features
    data_stats = data[features].describe()
    #print(data_stats)

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
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=1)
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
    models.append(('Dummy', DummyClassifier(strategy="constant",constant=1)))
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    #models.append(('KNN', KNeighborsClassifier(n_neighbors=1, metric='euclidean')))
    ##models.append(('CART', DecisionTreeClassifier()))
    ##models.append(('NB', GaussianNB()))
    models.append(('RF10', RandomForestClassifier(n_estimators=10,max_depth=2, random_state=0)))
    models.append(('RF50', RandomForestClassifier(n_estimators=50,max_depth=2, random_state=0)))
    #models.append(('RF300', RandomForestClassifier(n_estimators=300,max_depth=2, random_state=0)))
    
    ## CROSS VALIDATION ON TRAIN SET ----------------------------------------------
    # run train/test experiments
    cv_predictions, cv_performance = cross_validation(data_train, models, features, cv_splits=cv_splits,cv_repeats=cv_repeats)                  
    # display summary results
    performance_report = summarize_performance(cv_performance, threshold=0.5)
    print(performance_report)
    # plot mean Precision and Recall curves
    plot_PR_curves(cv_performance)
    plot_F_curves(cv_performance)

    ## FINAL EVALUATION ON TEST SET -----------------------------------------------
    print(' ')
    print('Final evaluation on test set:')
    print(' ')

    model_idx = 2
    model_name =  models[model_idx][0]
    model = models[model_idx][1] # RF50
    print(model)
    
    X_train = data_train[features] # features
    Y_train = data_train['class_ID'] #labels
    X_test = data_test[features] # features
    Y_test = data_test['class_ID'] #labels
    
    #print('WARNING !!!! TESTING ON TRAINING DATA')
    #X_test = X_train
    #Y_test = Y_train
    
    # feature normalization
    Norm_mean = X_train.mean()
    Norm_std = X_train.std()
    X_train = (X_train-Norm_mean)/Norm_std
    X_test = (X_test-Norm_mean)/Norm_std
        
    # Train on entire train set
    final_model = classification_train(X_train, Y_train, model)
    # Evaluate on full test set
    pred_class, pred_prob = classification_predict(X_test, final_model)
    # Print evaluation report
    CR = classification_report(Y_test, pred_class)
    print(CR)
    # save the model to disk
    model= {'name': model_name,
            'model':final_model,
            'features': features,
            'normalization_mean': Norm_mean,
            'normalization_std': Norm_std,
            'classes': class_encoder,
            }
    #pickle.dump(model, open(model_filename, 'wb'))

    # precision, recall, thresholds = precision_recall_curve(Y_val, pred_prob[:,0])
    # pr_auc = metrics.auc(recall, precision)
    # f1 = f1_score(Y_val, pred_class, average='binary')    
    # CR = classification_report(Y_val, pred_class)
    # CM = confusion_matrix(Y_val, pred_class)
    

if __name__ == "__main__":
    main()


