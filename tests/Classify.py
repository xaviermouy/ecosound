# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:51:43 2020

@author: xavier.mouy
"""
import pickle
model_filename = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Classification\LDA_model.sav'


# load the model from disk
loaded_model = pickle.load(open(model_filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)