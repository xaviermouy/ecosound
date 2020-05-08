# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:40:20 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import os
from ecosound.core.metadata import DeploymentInfo
from ecosound.core.annotation import Annotation

root_dir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019'
deployment_file = r'deployment_info.csv' 
annotation_dir = r'manual_annotations'
data_dir = r'audio_data'

# Instantiate
Deployment = DeploymentInfo()

# write empty file to fill in (do once only)
#Deployment.write_template(os.path.join(root_dir, deployment_file))

# load deployment file
deployment_info = Deployment.read(os.path.join(root_dir, deployment_file))

# load all annotations
annot = Annotation()
annot.from_raven(os.path.join(root_dir, annotation_dir), verbose=True)

# Mnaually fill in missing information
annot.insert_values(software_version='1.5',
                    operator_name='Emie Woodburn',
                    audio_channel=deployment_info.audio_channel_number[0],
                    UTC_offset=deployment_info.UTC_offset[0],
                    audio_file_dir=os.path.join(root_dir, data_dir),
                    audio_sampling_frequency=deployment_info.sampling_frequency[0],
                    audio_bit_depth=deployment_info.bit_depth[0],
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
print(' ')
print(annot.summary())

# save as parquet file
annot.to_parquet(os.path.join(root_dir, 'Annotations_dataset_' + deployment_info.deployment_ID[0] + '.parquet'))
annot.to_pamlab(root_dir, outfile='Annotations_dataset_' + deployment_info.deployment_ID[0] +' annotations.log', single_file=True)
annot.to_raven(root_dir, outfile='Annotations_dataset_' + deployment_info.deployment_ID[0] +'.Table.1.selections.txt', single_file=True)
