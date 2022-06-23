# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:35:34 2022

@author: xavier.mouy
"""

from pydoc import importfile
annotation_module = importfile(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\core\annotation.py")



#infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations'
infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections1.sqlite'
#infile = [r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections1.sqlite',r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections2.sqlite']

# load sqlite tables
annot = annotation_module.Annotation()
annot.from_sqlite(infile, verbose=True)


# write sqlite table
outfile =r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/write/detections3.sqlite'
annot.to_sqlite(outfile)

# load created sqlite tables
annot2 = annotation_module.Annotation()
annot2.from_sqlite(outfile, verbose=True)
