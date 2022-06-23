# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:35:34 2022

@author: xavier.mouy
"""

import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.
from ecosound.core.annotation import Annotation

#infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations'
infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections1.sqlite'
#infile = [r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections1.sqlite',r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections2.sqlite']

# load sqlite tables
annot = Annotation()
annot.from_sqlite(infile, verbose=True)


# write sqlite table
outfile =r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/write/detections3.sqlite'
annot.to_sqlite(outfile)

# load created sqlite tables
annot2 = Annotation()
annot2.from_sqlite(outfile, verbose=True)

agg_1D = annot2.calc_time_aggregate_1D(integration_time='1H',is_binary=True) 
agg_2D = annot2.calc_time_aggregate_2D(integration_time='1H',is_binary=False)