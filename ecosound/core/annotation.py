# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
import pandas as pd



class Annotation():
    """Defines an annotation object."""
   
    def __init__(self):
        self.data = pd.DataFrame({
            'UID':[],
            'from_detector': [], # True, False
            'software_name':[],
            'software_version':[],
            'operator_name':[],
            'UTC_offset':[],
            'entry_date':[],
            'audio_channel':[],
            'audio_file_name':[],
            'audio_file_dir':[],
            'audio_file_extension':[],
            'audio_file_start_date':[],
            'audio_sampling_frequency':[],
            'audio_bit_depth':[],
            'mooring_platform_name':[],
            'recorder_type':[],
            'recorder_SN':[],
            'hydrophone_model':[],
            'hydrophone_SN':[],
            'hydrophone_depth':[],
            'location_name':[],
            'location_lat':[],
            'location_lon':[],
            'location_water_depth':[],
            'offset_time_min':[],
            'offset_time_max':[],
            'duration':[],
            'frequency_min':[],
            'frequency_max':[],
            'date_min':[],
            'date_max':[],
            'label_source':[],
            'label_sound_type':[],
            'confidence':[]
            })



    def _str_to_list(file):
        """Convert str input to list if necessary."""
        if type(file) is str:
            file = [file]
        return file

    def read_raven(self, files):
        """import from 1 or several Raven files."""
    def to_raven(self, file):
        """write to a Raven files."""
    
    def read_pamlab(self, files):
        """import from 1 or several PAMLab files."""
        assert  type(files) in (str,list), "Input must be of type str (single file) or list (multiple files)"
        # Import files to dataframe
        df = pd.concat(map(lambda file: pd.read_csv(file, delimiter='\t'), files))
        #data = pd.read_csv(logfile, sep='\t', header=None, skiprows=1)
        #encoding='utf-8'
        return self
    
    def to_pamlab(self, file):
        """write to a PAMLab file."""        

    def read_pytable(self, conn):
        """import from pytable file."""
        
    def print():
        print('dsdsd')
            
    def __repr__(self):
        return 'Annotation object'




