# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:40:20 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.metadata import DeploymentInfo
from ecosound.core.annotation import Annotation

deployment_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\deployment_info.csv' 
annotation_dir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\manual_annotations'
data_dir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\audio_data'
# Instantiate
Deployment = DeploymentInfo()

# write empty file to fill in (do once only)
#Deployment.write_template(deployment_file)

# load deployment file
deployment_info = Deployment.read(deployment_file)

# load all annotations
annot = Annotation()
annot.from_raven(annotation_dir, verbose=True)

# Mnaually fill in missing information
annot.insert_values(software_version='1.5',
                    operator_name='Emie Woodburn',
                    UTC_offset=0,
                    audio_file_dir=data_dir,
                    audio_sampling_frequency=32000,
                    audio_bit_depth=32,
                    mooring_platform_name=deployment_info.mooring_platform_name[0],
                    recorder_type=deployment_info.recorder_type[0],
                    recorder_SN=deployment_info.recorder_SN[0],
                    hydrophone_model=deployment_info.hydrophone_model[0],
                    hydrophone_SN=deployment_info.hydrophone_SN[0],
                    hydrophone_depth=deployment_info.hydrophone_depth[0],
                    location_name = deployment_info.location_name[0],
                    location_lat = deployment_info.location_lat[0],
                    location_lon = deployment_info.location_lon[0],
                    location_water_depth = deployment_info.location_water_depth[0],
                    deployment_ID=deployment_info.deployment_ID[0],
                    )

# Correct species names where needed
print(annot.get_labels_class())
annot.data['label_class'].replace(to_replace=['FSFS',' FS'], value='FS', inplace=True)
annot.data['label_class'].replace(to_replace=['KW '], value='KW', inplace=True)
annot.data['label_class'].replace(to_replace=['Seal','Seal\\'], value='HS', inplace=True)
annot.data['label_class'].replace(to_replace=['Unknown','Chirp',' ','  '], value='UN', inplace=True)
annot.data['label_class'].dropna(axis=0, inplace=True)
print(annot.get_labels_class())

# print summary (pivot table)
#-> to implement
#annot.summary()

# save as parquet file




