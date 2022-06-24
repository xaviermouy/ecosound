# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:35:34 2022

@author: xavier.mouy
"""

import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.
from ecosound.core.annotation import Annotation
import ecosound

infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections1.sqlite'

# load sqlite tables
annot = Annotation()
annot.from_sqlite(infile, verbose=True)

# from ecosound.visualization.grapher_builder import GrapherFactory
# graph = GrapherFactory('AnnotHeatmap')
# graph.add_data(annot)
# graph.add_data(annot)
# graph.norm_value=[None, 20]
# graph.title='My title'
# graph.integration_time='60Min'
# graph.colormap='viridis'
# graph.is_binary=False
# graph.date_format = '%d-%b-%Y' #'%d-%b-%Y'
# graph.show()

#graph.to_file(r'C:\Users\xavier.mouy\Desktop\2Daggregate')

annot.heatmap(title='My title Yeah')



annot2 = Annotation()
annot2.to_netcdf(r'C:\Users\xavier.mouy\Desktop\asas')

import numpy as np
import pandas as pd

annot3 = Annotation()
annot3.to_raven(r'C:\Users\xavier.mouy\Desktop','test.Table.1.selections.txt')

# rows, cols = annot2.data.shape
# annot3.data = pd.DataFrame([[np.nan]*cols]*(rows+1),columns=annot2.data.columns)
# annot3.to_raven(r'C:\Users\xavier.mouy\Desktop','test.txt')

# outdf.to_csv('C:\Users\xavier.mouy\Desktop','test.txt',
#              sep='\t',
#              encoding='utf-8',
#              header=True,
#              columns=cols,
#              index=False)

# #nan1 = pd.DataFrame([[np.nan]*cols]*(rows+1),columns=annot2.data.columns)


