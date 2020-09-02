# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:34:43 2020

Load NetCDF files for an entire deployment and plot timeseries

@author: xavier.mouy
"""
import xarray

#indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_in_April_July2019_1342218252'
indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\test_dataset'

ds = xarray.open_mfdataset(indir + "/*.nc" ,parallel=True)
ds_label = ds.label_class
ds_conf = ds.confidence

ds.isel(date=10)
# f=ds.groupby('label_class').groups

# ds.sel(date="2019-04-11")