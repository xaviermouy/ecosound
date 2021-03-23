# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:24:27 2021

@author: xavier.mouy
"""
import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement

infile = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Herring_DFO\New folder\5042.200306203002.wav.nc'
outdir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Herring_DFO\New folder'

mes = Measurement()
mes.from_netcdf(infile)

mes.to_pamlab(outdir)
mes.to_raven(outdir)