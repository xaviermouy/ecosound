# -*- coding: utf-8 -*-.
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
# TODO: kjlkjkl;j

import pandas as pd
import os
import uuid
import ecosound.core.tools
import ecosound.core.decorators
from ecosound.core.annotation import Annotation


class Measurement(Annotation):

    def __init__(self):
        super(Measurement, self).__init__()
        self.measurer_name = None
        self.measurer_version = None
        self.measurements_name = None
        self.measurements_data = None

    def __add__(self, other):
        """Concatenate data from several annotation objects."""
        assert type(other) is ecosound.core.annotation.Annotation, "Object type not \
            supported. Can only concatenate Annotation objects together."
        self.data = pd.concat([self.data, other.data],
                              ignore_index=True,
                              sort=False)
        return self

    def __repr__(self):
        """Return the type of object."""
        return (f'{self.__class__.__name__} object ('
                f'{len(self.data)})')

    def __str__(self):
        """Return string when used with print of str."""
        return f'{len(self.data)} annotation(s)'

    def __len__(self):
        """Return number of annotations."""
        return len(self.data)
