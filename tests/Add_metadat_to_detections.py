# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:16:16 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
from ecosound.core.metadata import DeploymentInfo
import os
import ecosound.core.tools
import platform

outdir= r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Full_dataset_with_metadata'
ext='.nc'


indir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Full_dataset\UVIC_mill-bay_2019'
deployment_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_mill-bay_2019\deployment_info.csv'
data_dir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_mill-bay_2019\audio_data'

# load meta data
operator_name = platform.uname().node
dep_info = DeploymentInfo()
dep_info.read(deployment_file)

#list files
files = ecosound.core.tools.list_files(indir,
                                        ext,
                                        recursive=False,
                                        case_sensitive=True)

for idx,  file in enumerate(files):
    print(str(idx)+r'/'+str(len(files))+': '+ file)
    meas = Measurement()
    meas.from_netcdf(file)

    meas.insert_metadata(deployment_file)

    file_name = os.path.splitext(os.path.basename(file))[0]
    meas.insert_values(operator_name=platform.uname().node,
                       audio_file_name=os.path.splitext(os.path.basename(file_name))[0],
                       audio_file_dir=data_dir,
                       audio_file_extension='.wav',
                       audio_file_start_date= ecosound.core.tools.filename_to_datetime(file_name)[0]
                       )
    meas.to_netcdf(os.path.join(outdir,file_name+'.nc'))
