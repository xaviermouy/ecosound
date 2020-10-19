# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:03:39 2020

@author: xavier.mouy
"""
import xarray
import os
import numpy as np

#indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_In_Oct17_Dec3_2018_67674121'
indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_Out_April_July2019_1409343536' 

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

idx=0
D=np.empty(1)
for dirpath, dnames, fnames in os.walk(indir):
    for f in fnames:
        if f.endswith(".nc"):
            idx+=1
            print(f)
            ds = xarray.open_dataset(os.path.join(indir,f))
            ds = preprocess_func(ds)    
            Tstart = ds['date'].min().data
            Tend = ds['date'].max().data
            print(Tstart, Tend)            
            if idx==1:
                Tpast = Tstart - np.timedelta64(5,'ns')
            if Tpast >= Tstart:
                raise ValueError("ValueError")
            Tpast = Tend
            
            