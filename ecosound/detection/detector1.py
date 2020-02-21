# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:27:39 2020

@author: xavier.mouy
"""

from .detectors_builder import BaseClass

class Detector1(BaseClass):

    def __init__(self,*args,**kwargs):
        self.kwargs = kwargs
        self.args = args

    def who(self):
        print("Detector1")


