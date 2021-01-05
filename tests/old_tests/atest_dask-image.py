# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:07:09 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.decorators import timeit

#from dask.distributed import Client, LocalCluster
import dask_image.ndfilters
import dask.array
import numpy as np
from scipy import ndimage
import dask
from numba import njit

@njit
def calcVariance2D(buffer):
    """Calculate the 2D variance."""
    return np.var(buffer)

@njit
def calcMedian(buffer):
    """Calculate the 2D variance."""
    return np.median(buffer)

@timeit
def Filter_med_normal(array, window_duration):
    # # Apply filter
    Smed = ndimage.median_filter(array, (1, window_duration))
    return Smed

@timeit
def Filter_med_numba(array, window_duration):
    # # Apply filter
    Svar = ndimage.generic_filter(array, calcMedian,
                                  (1, window_duration),
                                  mode='mirror')
    return Svar

@timeit
def Filter_var_normal(array,kernel_bandwidth, kernel_duration):
    # # Apply filter
    Svar = ndimage.generic_filter(array, calcVariance2D,
                                  (kernel_bandwidth, kernel_duration),
                                  mode='mirror')
    return Svar
@timeit
def Filter_var_dask(array,kernel_bandwidth, kernel_duration):
    dask_spectro = dask.array.from_array(array, chunks=(1000,1000))
    Svar = dask_image.ndfilters.generic_filter(dask_spectro,
                                               calcVariance2D,
                                               size=(kernel_bandwidth, kernel_duration),
                                               mode='mirror')
    Svar = Svar.compute()
    return Svar

@timeit
def Filter_med_dask(array, window_duration):
    dask_spectro = dask.array.from_array(array, chunks=(2048, 1000))
    Svar = dask_image.ndfilters.median_filter(dask_spectro,
                                               size=(1, window_duration),
                                               mode='mirror')
    Svar = Svar.compute()
    return Svar

# # Setup a local cluster.
# # By default this sets up 1 worker per core
# cluster = LocalCluster()
# client = Client(cluster)


array = np.random.rand(2048, 200000)

## Variance filter
# kernel_bandwidth = 100
# kernel_duration = 20
# S1 = Filter__var_normal(array,kernel_bandwidth, kernel_duration)
# S2 = Filter_var_dask(array, kernel_bandwidth, kernel_duration)

## Median filter
window_duration = 10
#Smed1 = Filter_med_normal(array, window_duration)
#Smed2 = Filter_med_numba(array, window_duration)
Smed3 = Filter_med_dask(array, window_duration)