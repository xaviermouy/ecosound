# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:06:18 2020

@author: xavier.mouy
"""
import xarray as xr
import os

indir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run2\RCA_in_April_July2019_1342218252'
file_ext = '.nc'
          
# loops thrugh each file
error_list =[]
files_list = os.listdir(indir) # list of files
for file in files_list:
    if os.path.isfile(os.path.join(indir, file)) & file.endswith(file_ext):
        try:
            ss = xr.open_dataset(os.path.join(indir,file))
        except:
            error_list.append(file)
                   