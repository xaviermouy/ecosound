# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:34:43 2020

Load NetCDF files for an entire deployment and plot timeseries

@author: xavier.mouy
"""
import xarray


def xr_sort(ds):
    return ds.sortby(ds.time_min_date,ascending=True)

#indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_in_April_July2019_1342218252'
indir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\test_dataset'

# load data
ds = xarray.open_mfdataset(indir + "/*.nc" ,parallel=True, preprocess=xr_sort)

# sort by ascending date
#ds = ds.sortby(ds.time_min_date,ascending=True)
# filter by class and confidence
#ds_filtered = ds.where(ds.label_class == 'FS').where(ds.confidence>0.7).compute()
#ds_filtered = ds.where((ds.label_class == 'FS') & (ds.confidence>0.7), drop=True)

# Hourly count
#fs_count = ds_filtered.groupby('date.hour').count(dim='date') 
#fs_count2 = ds_filtered.resample(date='H').count(dim='date') 
fs_hourly_count = ds.where((ds.label_class == 'FS') & (ds.confidence>0.7), drop=True).resample(date='H').count(dim='date')
fs_hourly_count.compute()
# plot
fs_hourly_count.label_class.plot()

## min confidence per day
#ds_conf_daily = ds_filtered.confidence.resample(date='D').min(dim='date')
#ds_conf_daily.plot()

#s=[ds_filtered['label_class'].isel(date=slice(0,-1))]

#ds.sortby(time_min_date,ascending=True)

#ds_label = ds.label_class
#ds_conf = ds.confidence

#ds_conf.sortby(date, ascending=True)
#ds_conf_daily = ds_conf.resample(time='D').mean(dim='date')

#ds_conf.mean().compute()

#ds_label.set_index(data["Time"],inplace=True)
# f=ds.groupby('label_class').groups

# ds.sel(date="2019-04-11")