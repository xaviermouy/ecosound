# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:28:29 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement


# # ## netcdf folder
# netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test'
# annot4 = Annotation()
# annot4.from_netcdf(netcdf_files, verbose=True)
# print(len(annot4))


# ## Load netcdf measurmeent folder from folder
netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2'
meas = Measurement()
meas.from_netcdf(netcdf_files, verbose=True)
print(len(meas))

# # ## Load netcdf measurmeent folder from single file
# netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2\67391492.181017121114.wav.nc'
# meas = Measurement()
# meas.from_netcdf(netcdf_files, verbose=True)
# print(len(meas))

# # ## Load netcdf measurmeent folder from list of files
# netcdf_files = []
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2\67391492.181017121114.wav.nc")
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2\67391492.181017151114.wav.nc")
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2\67391492.181017181114.wav.nc")
# meas = Measurement()
# meas.from_netcdf(netcdf_files, verbose=True)
# print(len(meas))


