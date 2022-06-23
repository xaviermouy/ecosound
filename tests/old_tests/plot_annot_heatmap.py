# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:35:34 2022

@author: xavier.mouy
"""

from pydoc import importfile
annotation_module = importfile(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\core\annotation.py")



infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\sqlite_annotations/read/detections1.sqlite'

# load sqlite tables
annot = annotation_module.Annotation()
annot.from_sqlite(infile, verbose=True)

from ecosound.visualization.grapher_builder import GrapherFactory
graph = GrapherFactory('HeatmapPlotter')