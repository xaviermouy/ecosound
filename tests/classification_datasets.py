# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:24:56 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
from ecosound.core.annotation import Annotation

meas_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset.nc'
annot_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\Master_annotations_dataset.nc'

annot = Annotation()
annot.from_netcdf(annot_file)

nfold = 10
make_balanced = False
stratification=['label_class','recorder type','deployment_ID']
groups = hour

balanced = True

# > Add group
# startification

GroupKFold 
 



