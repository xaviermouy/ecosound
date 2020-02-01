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
        
    def check_integrity(self, verbose=False, time_duplicates_only=False):
        """Check integrity of Annotation object.
        
        Check: start/stop times, min/max frequencies
        Remove: duplicate entries        
        """
        # Drop all duplicates
        count_start = len(self.data)
        if time_duplicates_only: # remove annot with exact same time boundaries
            self.data = self.data.drop_duplicates(subset=['time_min_offset',
                                                    'time_max_offset',
                                                    'label_class',
                                                    'label_subclass',
                                                    'audio_file_name',
                                                    ], keep = "first",
                                                  ).reset_index(drop=True)
        else:  # remove annot with exact same time AND frequency boundaries
            self.data = self.data.drop_duplicates(subset=['time_min_offset',
                                                    'time_max_offset',
                                                    'frequency_min',
                                                    'frequency_max',
                                                    'label_class',
                                                    'label_subclass',
                                                    'audio_file_name',
                                                    ], keep = "first",
                                                  ).reset_index(drop=True)
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

    def from_raven(self, files, class_header='Sound type', subclass_header=None, verbose=False):
        """Import from 1 or several Raven files."""
        data = Annotation._import_files(files)
        files_timestamp = core.tools.filename_to_datetime(data['Begin Path'].tolist())
        self.data['audio_file_start_date'] = files_timestamp
        self.data['audio_channel'] = data['Channel']
        self.data['audio_file_name'] = data['Begin Path'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        self.data['audio_file_dir'] = data['Begin Path'].apply(
            lambda x: os.path.dirname(x))
        self.data['audio_file_extension'] = data['Begin Path'].apply(
            lambda x: os.path.splitext(x)[1])
        self.data['time_min_offset'] = data['Begin Time (s)']
        self.data['time_max_offset'] = data['End Time (s)']
        self.data['time_min_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] + pd.to_timedelta(self.data['time_min_offset'], unit='s'))
        self.data['time_max_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] +
            pd.to_timedelta(self.data['time_max_offset'], unit='s'))
        self.data['frequency_min'] = data['Low Freq (Hz)']
        self.data['frequency_max'] = data['High Freq (Hz)']
        
        if class_header is not None:
            self.data['label_class'] = data[class_header]
        if subclass_header is not None:
            self.data['label_subclass'] = data[subclass_header]
        self.data['from_detector'] = False
        self.data['software_name'] = 'raven'
        self.data['uuid'] = self.data.apply(lambda _: uuid.uuid4(), axis=1)
        self.data['duration'] = self.data['time_max_offset'] - self.data['time_min_offset']
        self.check_integrity(verbose=verbose,time_duplicates_only=True)

    def to_raven(self, file):
        """Write to a Raven files."""

    def from_pamlab(self, files, verbose=False):
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
        self.check_integrity(verbose=verbose)
    
    def get_labels_class(self):
        """Return all unique class labels."""
        if len(self.data)>0:
            classes = list(self.data['label_class'].unique())
        else:
            classes = []
        return classes

    def get_labels_subclass(self):
        """Return all unique class labels."""
        if len(self.data)>0:
            subclasses = list(self.data['label_subclass'].unique())
        else:
            subclasses = []
        return subclasses


    @staticmethod 
    @core.decorators.listinput
    def _import_files(files):
        """Import one or several text files with header to a Panda datafrane."""
        assert  type(files) in (str,list), "Input must be of type str (single file) or list (multiple files)"
        # Import all files to a dataframe
        for idx, file in enumerate(files):
            # Extract header first due to formating issues in PAMlab files
            header = pd.read_csv(file,
                                 delimiter = '\t',
                                 header=None,
                                 nrows=1)
            headerLength = header.shape[1]
            # Get all data and only keep values correpsonding to header labels
            tmp = pd.read_csv(file,
                              delimiter = '\t',
                              header=None,
                              skiprows=1,
                              na_values=None)
            tmp = tmp.iloc[:,0:headerLength]
            # Put header back
            tmp = tmp.set_axis(list(header.values[0]), axis=1, inplace=False)
            if idx == 0:
                data = tmp
            else:
                data= pd.concat([data,tmp], ignore_index=True,sort=False)        
        return data

    def to_pamlab(self, outdir):
        """Write to a PAMLab file."""        
        annot = self.data
        
        cols = ['fieldkey:', 'Soundfile', 'Channel', 'Sampling freq (Hz)', 'Latitude (deg)', 'Longitude (deg)', 'Recorder ID', 'Recorder depth', 'Start date and time (UTC)', 'Annotation date and time (local)', 'Recorder type', 'Deployment', 'Station', 'Operator', 'Left time (sec)', 'Right time (sec)', 'Top freq (Hz)', 'Bottom freq (Hz)', 'Species', 'Call type', 'rms SPL', 'SEL', '', '']
        outdf = pd.DataFrame({'fieldkey:': 0, 'Soundfile': 0, 'Channel': 0, 'Sampling freq (Hz)': 0, 'Latitude (deg)': 0, 'Longitude (deg)': 0, 'Recorder ID': 0, 'Recorder depth': 0, 'Start date and time (UTC)': 0, 'Annotation date and time (local)': 0, 'Recorder type': 0, 'Deployment': 0, 'Station': 0, 'Operator': 0, 'Left time (sec)': 0, 'Right time (sec)': 0, 'Top freq (Hz)': 0, 'Bottom freq (Hz)': 0, 'Species': 0, 'Call type': 0, 'rms SPL': 0, 'SEL': 0, '': 0, '': 0}, index=list(range(self.output.shape[0])))    
        outdf['fieldkey:'] = 'an:'
        outdf['Species'] = annot['label_class']
        outdf['Call type'] = annot['label_subclass']
        outdf['Left time (sec)'] = annot['time_min_offset']
        outdf['Right time (sec)'] = annot['time_max_offset']
        outdf['Top freq (Hz)'] = annot['frequency_max']
        outdf['Bottom freq (Hz)'] = annot['frequency_min']
        outdf['rms SPL'] = annot['confidence']
        outdf['Operator'] =annot['operator_name']
        outdf['Channel'] =annot['channel']
        if len(annot.fileName) == 0:
            outdf['Soundfile'] = os.path.join(str(annot.filePath[0]), str(annot.fileName[0])) + str(annot.fileExtension[0])
        else:
            filenames=[]
            for i in range(0,len(annot.fileName)):
                filenames.append(os.path.join(str(annot.filePath[i]), str(annot.fileName[i])) + str(annot.fileExtension[i]))
            outdf['Soundfile'] = filenames             
        annot.to_csv(os.path.join(outdir, str(outdf.fileName[0])) + str(outdf.fileExtension[0]) + ' chan' +  str(outdf.channel[0]) + ' annotations.log', sep='\t', encoding='utf-8', header=True, columns=cols, index=False)

    def from_pytable(self, conn):
        """Import from pytable file."""

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



