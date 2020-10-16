# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:38:07 2020

@author: xavier.mouy
"""
import xarray
import numpy as np
import os
import time

def xr_sort(ds):
    return ds.sortby(ds.time_min_date,ascending=True)

def to_drop_vars(ds):
    return [x for x in list(ds.variables) if x not in ['date','time_min_date','label_class','label_subclass','confidence']] 

def preprocess_func(ds):
    return xr_sort(ds.drop(to_drop_vars(ds))) 

indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_in_April_July2019_1342218252'
#indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\test_dataset'
outfile ='hourly_summary.nc'
confidence_step = 0.02

# Start timer
tic = time.perf_counter()

# load data
print('')
print('Loading dataset')
tic1 = time.perf_counter()
ds = xarray.open_mfdataset(indir + "/*.nc" ,
                           parallel=True,
                           preprocess=preprocess_func,
                           data_vars=['time_min_date','label_class','label_subclass','confidence'],
                           coords=['date'],
                           )
toc1 = time.perf_counter()
print(f"Executed in {toc1 - tic1:0.4f} seconds")

# Extract hourly summaries for each class and threshold
print('')
print('Calculating hourly detections')
tic2 = time.perf_counter()
class_groups = ds.groupby('label_class')
confidence_thresholds = np.arange(1/len(class_groups),1,confidence_step)
class_names = []
Xarrays = []
for name, group in class_groups:
    print('  * Extracting class: ' + name)
    class_names.append(name)
    max_threshold_reached = False
    for conf_idx, threshold in enumerate(confidence_thresholds):
        if max_threshold_reached == False: # 
            detec_thresholded=group.where((group.confidence>=threshold),drop=True)
            if len(detec_thresholded['date'])>0: # if there are detection for that threshold
                detec = detec_thresholded.resample(date='H').count(dim='date')
                print('    - confidence: ', '%.2f'%threshold)
                if conf_idx == 0:
                    # define date array
                    date_array = detec['date']
                    # initialize detec array with nan
                    detec_array = np.empty([len(confidence_thresholds),detec.dims['date']])
                    detec_array[:] = np.NaN            
                # fill in appropriate row of detec array with detection counts
                detec_array[conf_idx,:] = detec['label_class'].values
            else: # if there are no detections anymore
                # fill in appropriate row of detec array with zeros
                detec_array[conf_idx,:] = 0
                max_threshold_reached = True
                print('    - confidence: ', '%.2f'%threshold, ' (empty)')
        else: # fills with zeros once max confidence threshold is reached
            # fill in appropriate row of detec array with zeros
            detec_array[conf_idx,:] = 0
            print('    - confidence: ', '%.2f'%threshold, ' (empty)')
    #create xArray for that class
    Xarrays.append(xarray.DataArray(detec_array, coords={'date': date_array, 'confidence': confidence_thresholds}, dims=['confidence','date']))
#Create XArrayDataset
dataset = xarray.Dataset(dict(zip(class_names,Xarrays)))
# Save dataset as netcdf
outdir = os.path.join(indir,'summary')
os.mkdir(outdir)
dataset.to_netcdf(os.path.join(outdir,outfile))
toc2 = time.perf_counter()
print(f"Executed in {toc2 - tic2:0.4f} seconds")

# Stop timer
toc = time.perf_counter()
print('')
print(f" Overall run time is {toc - tic:0.4f} seconds")