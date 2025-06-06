# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:38:07 2020

@author: xavier.mouy
"""
import xarray
import numpy as np
import os
import time
import datetime as dt
import argparse

def xr_sort(ds):
    #return ds.sortby(ds.time_min_date,ascending=True)
    ds = ds.sortby(ds.date,ascending=True)
    
    return ds.sortby(ds.date,ascending=True)

def to_drop_vars(ds):
    return [x for x in list(ds.variables) if x not in ['date','audio_file_start_date','time_min_date','label_class','label_subclass','confidence']] 

def fix_duplicates(ds):
    index=range(0,len(ds['date']))
    _, unique_index = np.unique(ds['date'], return_index=True)
    index = np.delete(index,unique_index)
    if len(index)>0:
        rnd = np.round(np.random.rand(len(index))*100)-50
        rnd[rnd==0]=1
        rnd_offset =[np.timedelta64(int(x),'ns') for x in rnd]
        time_ax = ds['date'].data
        time_ax[np.array(index)] = time_ax[index] + rnd_offset
        ds['date'] = time_ax
    #ds.time_min_date = time_ax
    return ds

def shift_max_dates(ds,files_dur_sec):
    time_ax = ds['date'].data    
    max_date = ds['audio_file_start_date'][0].data + np.timedelta64(files_dur_sec,'s')
    index = time_ax >= max_date
    time_ax[np.array(index)] = max_date - np.timedelta64(1,'ns')    
    ds['date'] = time_ax    
    return ds

def preprocess_func(ds):
    files_dur_sec = 1799
    return shift_max_dates(xr_sort(fix_duplicates(ds.drop(to_drop_vars(ds)))),files_dur_sec) 



parser = argparse.ArgumentParser(description="Create detections hourly summary")

# define command line arguments
parser.add_argument('--detec_folder', type=str, default=None,
                        help='path to the folder containing the .nc files')
parser.add_argument('--confidence_step', type=float, default=None,
                        help='Confidence steps between >0 and <1 ')

args = parser.parse_args()


if args.detec_folder:
    indir = args.detec_folder
else:
    #indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_Out_Jan_April2019_67391492'
    indir=r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ'
    #indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\test_dataset'

if args.confidence_step:
       confidence_step = args.confidence_step
else:
    confidence_step = 0.10

outfile ='hourly_summary.nc'


# Start timer
tic = time.perf_counter()

# load data
print('')
print('Loading dataset')
tic1 = time.perf_counter()
# ds = xarray.open_mfdataset(indir + "/*.nc" ,
#                             parallel=True,
#                             #preprocess=preprocess_func,
#                             data_vars=['time_min_date','audio_file_start_date','label_class','label_subclass','confidence'],
#                             coords=['date'],
#                             #combine='nested',
#                             #join='outer',
#                             )
ds = xarray.open_mfdataset(indir + "/*.nc",                           
                           chunks=10,
                           #concat_dim='date',
                           data_vars=['time_min_date','audio_file_start_date','label_class','label_subclass','confidence'],
                           coords='minimal',#['date'],
                           parallel=False,
                           )
toc1 = time.perf_counter()
print(f"Executed in {toc1 - tic1:0.4f} seconds")


#ds = xarray.open_dataset(r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\Out_Jan_April2019.nc')

# Extract hourly summaries for each class and threshold
print('')
print('Calculating hourly detections')
tic2 = time.perf_counter()
class_groups = ds.groupby('label_class')
if len(class_groups)==1:
    threshold_min = 0.5
else:
    threshold_min = 1/len(class_groups)
confidence_thresholds = np.arange(threshold_min,1,confidence_step)
class_names = []
Xarrays = []
for name, group in class_groups:
    print('  * Extracting class: ' + name)
    class_names.append(name)
    max_threshold_reached = False
    for conf_idx, threshold in enumerate(confidence_thresholds):
        if max_threshold_reached == False: # 
            detec_thresholded=group.where((group.confidence>=threshold),drop=False)
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