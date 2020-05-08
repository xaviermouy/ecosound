# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:40:20 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import os
from ecosound.core.annotation import Annotation

root_dir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets'
dataset_files = ['UVIC_mill-bay_2019\Annotations_dataset_06-MILL.parquet',
                 'UVIC_hornby-island_2019\Annotations_dataset_07-HI.parquet',
                 'ONC_delta-node_2014\Annotations_dataset_ONC-Delta-2014.parquet',
                 'DFO_snake-island_rca-in_20181017\Annotations_dataset_SI-RCAIn-20181017.parquet',
                 'DFO_snake-island_rca-out_20181015\Annotations_dataset_SI-RCAOut-20181015.parquet',
                ]

# # load all annotations
annot = Annotation()
for file in dataset_files:
    tmp = Annotation()
    tmp.from_parquet(os.path.join(root_dir, file), verbose=True)
    annot = annot + tmp

# print summary (pivot table)
print(' ')
print(annot.summary())

# save as parquet file
annot.to_parquet(os.path.join(root_dir, 'Master_annotations_dataset.parquet'))

