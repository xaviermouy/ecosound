# -*- coding: utf-8 -*-:
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
import pandas as pd
import os
import uuid
import core.tools
import core.decorators
import core.tools

class Annotation():
    """Defines an annotation object."""
   
    def __init__(self):
        self.data = pd.DataFrame({
            'uuid':[],
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
            'frequency_min':[],
            'frequency_max':[],
            'time_min_offset':[],
            'time_max_offset':[],
            'time_min_date':[],
            'time_max_date':[],
            'duration':[],
            'label_class':[],
            'label_subclass':[],
            'confidence':[]
            })
        
    def check_integrity(self, verbose=False):
        """Check integrity of Annotation object.
        
        Check: start/stop times, min/max frequencies
        Remove: duplicate entries        
        """
        # Drop all duplicates
        count_start = len(self.data)
        self.data = self.data.drop_duplicates(subset=['time_min_offset',
                                                'time_max_offset',
                                                'frequency_min',
                                                'frequency_max',
                                                'label_class',
                                                'label_subclass',
                                                'audio_file_name',
                                                ], keep = "first",)
        count_stop = len(self.data)
        if verbose:
            print('Duplicate entries removed:', str(count_start-count_stop))
        # Check that start and stop times are coherent (i.e. t2 > t1)
        time_check= self.data.index[
            self.data['time_max_offset']<self.data['time_min_offset']].tolist()
        if len(time_check) > 0:
            raise ValueError('Incoherent annotation times (time_min > time_max). Problematic annotations:' + str(time_check))
        # Check that min and max frequencies are coherent (i.e. fmin < fmax)
        freq_check= self.data.index[
            self.data['frequency_max']<self.data['frequency_min']].tolist()
        if len(freq_check) > 0:
            raise ValueError('Incoherent annotation frequencies (frequency_min > frequency_max). Problematic annotations:' + str(freq_check))
        if verbose:
            print('Integrity test succesfull')

    def from_raven(self, files):
        """Import from 1 or several Raven files."""
    def to_raven(self, file):
        """Write to a Raven files."""
    
    def from_pamlab(self, files):
        """Import from 1 or several PAMLab files."""
        data = Annotation._import_files(files)
        files_timestamp = core.tools.filename_to_datetime(data['Soundfile'].tolist())
        self.data['audio_file_start_date'] = files_timestamp
        self.data['operator_name'] = data['Operator']
        self.data['entry_date'] = pd.to_datetime(data['Annotation date and time (local)'], format='%Y-%m-%d %H:%M:%S.%f')
        self.data['audio_channel'] = data['Channel']
        self.data['audio_file_name'] = data['Soundfile'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        self.data['audio_file_dir'] = data['Soundfile'].apply(
            lambda x: os.path.dirname(x))
        self.data['audio_file_extension'] = data['Soundfile'].apply(
            lambda x: os.path.splitext(x)[1])
        self.data['audio_sampling_frequency'] = data['Sampling freq (Hz)']
        self.data['recorder_type'] = data['Recorder type']
        self.data['recorder_SN'] = data['Recorder ID']
        self.data['hydrophone_depth'] = data['Recorder depth']
        self.data['location_name'] = data['Station']
        self.data['location_lat'] = data['Latitude (deg)']
        self.data['location_lon'] = data['Longitude (deg)']
        self.data['time_min_offset'] = data['Left time (sec)']
        self.data['time_max_offset'] = data['Right time (sec)']

        self.data['time_min_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] + pd.to_timedelta(self.data['time_min_offset'], unit='s'))
            
        self.data['time_max_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] +
            pd.to_timedelta(self.data['time_max_offset'], unit='s'))

        self.data['frequency_min'] = data['Bottom freq (Hz)']
        self.data['frequency_max'] = data['Top freq (Hz)']
        self.data['label_class'] = data['Species']
        self.data['label_subclass'] = data['Call type']
        self.data['from_detector'] = False
        self.data['software_name'] = 'pamlab'
        self.data['uuid'] = self.data.apply(lambda _: uuid.uuid4(), axis=1)
        self.data['duration'] = self.data['time_max_offset'] - self.data['time_min_offset']
        self.check_integrity(verbose=False)

    @staticmethod 
    @core.decorators.listinput
    def _import_files(files):
        """Import one or several text files with header to a Panda datafrane."""
        assert  type(files) in (str,list), "Input must be of type str (single file) or list (multiple files)"
        # Import all files to a dataframe
        for idx, file in enumerate(files):
            # Extract header first due to formating issues in PAMlab files
            header = pd.read_csv(file, delimiter = '\t',header=None, nrows=1)
            headerLength = header.shape[1]
            # Get all data and only keep values correpsonding to header labels
            tmp = pd.read_csv(file, delimiter = '\t',header=None, skiprows=1)
            tmp = tmp.iloc[:,0:headerLength]
            # Put header back
            tmp = tmp.set_axis(list(header.values[0]), axis=1, inplace=False)
            if idx == 0:
                data = tmp
            else:
                data= pd.concat([data,tmp], ignore_index=True)        
        return data


               

    def to_pamlab(self, file):
        """Write to a PAMLab file."""        

    def from_pytable(self, conn):
        """Import from pytable file."""
    @property
    def sound_class_labels(self):
        """Return the set of unique sound class labels."""
        return set(self.data.sound_class_label)

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



