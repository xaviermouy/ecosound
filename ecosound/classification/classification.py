# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:21:12 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
#from ecosound.measurements.measurer_builder import MeasurerFactory
import pickle
import numpy as np

class Classifier():
    
    def __init__(self):
        self.model = None
        self.features = None
        self.Norm_mean = None
        self.Norm_std = None
        self.classes_encoder = None        
        
    def load_model(self, model_file):
        # laod classif model      
        classif_model = pickle.load(open(model_file, 'rb'))  
        self.features = classif_model['features']
        self.model = classif_model['model']
        self.Norm_mean = classif_model['normalization_mean']
        self.Norm_std = classif_model['normalization_std']
        self.classes_encoder = classif_model['classes']
        
    def classify(self, measurements):
        # data dataframe
        data = measurements.data
        n1=len(data)
        # drop observations/rows with NaNs
        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(subset=self.features, axis=0, how='any', thresh=None, inplace=True)
        n2=len(data)
        # Classification - predictions
        X = data[self.features]
        X = (X-self.Norm_mean)/self.Norm_std
        pred_class = self.model.predict(X)
        pred_prob = self.model.predict_proba(X)
        pred_prob = pred_prob[range(0,len(pred_class)),pred_class]
        # Relabel
        for index, row in self.classes_encoder.iterrows():
            pred_class = [row['label'] if i==row['ID'] else i for i in pred_class]
        # update measurements
        data['label_class'] = pred_class
        data['confidence'] = pred_prob
        measurements.data = data
        return measurements
            