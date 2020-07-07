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


# # ## Load netcdf measurmeent folder from folder
# netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Noise_dataset'
# outfile=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Noise_dataset\dataset_noise.nc'
# meas = Measurement()
# meas.from_netcdf(netcdf_files, verbose=True)
# print(len(meas))
# #meas.to_netcdf(outfile)

# # ## Load netcdf measurmeent folder from single file
netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Full_dataset_with_metadata2\JASCOAMARHYDROPHONE742_20140913T115018.797Z.wav.nc'
meas = Measurement()
meas.from_netcdf(netcdf_files, verbose=True)
print(len(meas))

# # ## Load netcdf measurmeent folder from list of files
# netcdf_files = []
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2\67391492.181017121114.wav.nc")
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2\67391492.181017151114.wav.nc")
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2\67391492.181017181114.wav.nc")
# meas = Measurement()
# meas.from_netcdf(netcdf_files, verbose=True)
# print(len(meas))


