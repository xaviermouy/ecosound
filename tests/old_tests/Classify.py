# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:51:43 2020

@author: xavier.mouy
"""
import pickle
model_filename = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Classification\LDA_model.sav'
import matplotlib.pyplot as plt
import numpy as np

# load the model from disk
#loaded_model = pickle.load(open(model_filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

y=np.random.rand(100,1)
x=range(0,100,1)
fig, ax = plt.subplots(1,1)
ax.plot(x,y)

#style = dict(size=10, color='gray')
bbox_props = dict(boxstyle="square", fc="w", ec="w", alpha=0.5)
ax.text(1, 1, "New Year's Day", size=5, bbox=bbox_props)