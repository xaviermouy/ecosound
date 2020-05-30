# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:28:29 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
from ecosound.core.annotation import Annotation


# PAMlab_files = []
# Raven_files = []
# Raven_files.append(r".\resources\AMAR173.4.20190916T061248Z.Table.1.selections.txt")
# annot1 = Annotation()
# annot1.from_raven(Raven_files, verbose=False)
# print(len(annot1))

# Raven_files = []
# Raven_files.append(r".\resources\67674121.181018013806.Table.1.selections.txt")
# annot2 = Annotation()
# annot2.from_raven(Raven_files, verbose=False)
# print(len(annot2))

# PAMlab_files = []
# PAMlab_files.append(r"..\ecosound\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log")
# annot3 = Annotation()
# annot3.from_pamlab(PAMlab_files, verbose=False)
# print(len(annot3))

# annot = annot1 + annot2 + annot3
# print(len(annot))

# print(annot.get_fields())
# annot.insert_values(operator_name='Xavier Mouy')


# ## netcdf 1 file only
# netcdf_files = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\DFO_snake-island_rca-in_20181017\Annotations_dataset_SI-RCAIn-20181017.nc"
# annot4 = Annotation()
# annot4.from_netcdf(netcdf_files, verbose=True)
# print(len(annot4))


# # ## netcdf several files
# netcdf_files = []
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\DFO_snake-island_rca-in_20181017\Annotations_dataset_SI-RCAIn-20181017.nc")
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\DFO_snake-island_rca-out_20181015\Annotations_dataset_SI-RCAOut-20181015.nc")
# netcdf_files.append(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\ONC_delta-node_2014\Annotations_dataset_ONC-Delta-2014.nc")
# annot4 = Annotation()
# annot4.from_netcdf(netcdf_files, verbose=True)
# print(len(annot4))

# # ## netcdf folder
# netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test'
# annot4 = Annotation()
# annot4.from_netcdf(netcdf_files, verbose=True)
# print(len(annot4))

# # ## netcdf folder from Measurements
# netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2'
# annot4 = Annotation()
# annot4.from_netcdf(netcdf_files, verbose=True)
# print(len(annot4))

# ## netcdf folder from Measurements
netcdf_files = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\test2'
annot4 = Measurement()
annot4.from_netcdf(netcdf_files, verbose=True)
print(len(annot4))

# import xarray as xr
# d=annot3.data
# index = range(0,len(d),1)
# d['index']=index
# #d = d.set_index(['index','entry_date', 'frequency_min','label_class'])
# d = d.set_index(['index'])

# data = d.to_xarray()

# data2=data.sel(index=0)