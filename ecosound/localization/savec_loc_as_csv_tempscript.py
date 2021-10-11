# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:27:05 2021

@author: xavier.mouy
"""
import os
import sys
sys.path.append(r'C:\Users\xavier.mouy\Documents\GitHub\ecosound') # Adds higher directory to python modules path.

import pandas as pd
from ecosound.core.measurement import Measurement

indir = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mobile_array_copper'

# load data
print('')
print('Loading dataset')
idx = 0
for infile in os.listdir(indir):
    if infile.endswith(".nc"):
        print(infile)
        
        locs = Measurement()
        locs.from_netcdf(os.path.join(indir,infile))
        loc_data = locs.data
        if idx == 0:
            final_df = loc_data
        else:
            final_df = final_df.append(loc_data,ignore_index=True)        
        idx +=1

final_df.to_csv(os.path.join(indir,'localizations_python.csv'))