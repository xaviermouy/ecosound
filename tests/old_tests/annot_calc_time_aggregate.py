# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:35:34 2022

@author: xavier.mouy
"""

import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.
from ecosound.core.annotation import Annotation

infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections1.sqlite'

# load sqlite tables
annot = Annotation()
annot.from_sqlite(infile, verbose=True)

# calculate aggregates
agg_1D = annot.calc_time_aggregate_1D(integration_time='1H',is_binary=True) 
agg_2D = annot.calc_time_aggregate_2D(integration_time='1H',is_binary=False)