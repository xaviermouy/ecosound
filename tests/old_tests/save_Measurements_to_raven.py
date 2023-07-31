# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:22:20 2023

@author: xavier.mouy
"""
from ecosound.core.measurement import Measurement


infile = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\results\AMAR173.1.20220821T180710Z.wav.nc'
outdir = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\results'

loc = Measurement()
loc.from_netcdf(infile)

loc.to_raven(outdir)