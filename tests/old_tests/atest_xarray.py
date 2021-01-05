# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:43:58 2020

@author: xavier.mouy
"""


import xarray as xr
import pandas as pd
import numpy as np

# # Test 1: test if xarray and netcdf can hav index with same values
# df=pd.DataFrame()
# df['index']=[1,2,3,4,4,5,6,7,8,8,9,10]
# df['value1']=np.random.rand(12)
# df['value2']=np.random.rand(12)
# d=df.set_index(['index'])
# dxr=d.to_xarray()
# dxr.to_netcdf('test1.nc')
# dxr.sel(index=4)
# print('Show the 2 values with index value of 4')
# print(float(dxr.sel(index=4)['value1'][0]))
# print(float(dxr.sel(index=4)['value1'][1]))
# print('Show the 2 values with index value of 4 after writing and reimporting the xarray to netcdf')
# dxr2 = xr.open_dataset('test1.nc')
# print(float(dxr2.sel(index=4)['value1'][0]))
# print(float(dxr2.sel(index=4)['value1'][1]))

# Test 2: test if xarray csan store matrix
df=pd.DataFrame()
matrix = [np.random.rand(40,40)]*12

df = pd.DataFrame({'index':[1,2,3,4,4,5,6,7,8,8,9,10],
                   'value1': np.random.rand(12),
                   'value2': np.random.rand(12),
                   'matrix': matrix,
                   #'matrix': [],
                   })
d=df.set_index(['index'])
print('Value panda before converting to xarray')
print(d.head())
print('Type of first matrix: ',str(type(d['matrix'].iloc[0])))
# convert to xarray and extract :
dxr=d.to_xarray()
mat = dxr.sel(index=1)['matrix'].values
print('same Value from xarray ')
print(mat)
print('type: ',type(mat))
dxr.to_netcdf('test2.nc')
#dxr2 = xr.open_dataset('test2.nc')
#dxr2.sel(index=1)['matrix'].values

# Test 3: test if 2 xarrays from annot and measurements can be combined into 1
