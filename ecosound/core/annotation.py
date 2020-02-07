# -*- coding: utf-8 -*-.
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
# TODO: kjlkjkl;j

import pandas as pd
import os
import uuid
import core.tools
import core.decorators
import core.tools


class Annotation():
    """
    A class used for manipulating annotation data.

    The Annotation object stores from both manual analysis annotations
    collected with software like PAMlab and Raven, and outputs from
    automated detectors and classifiers.

    Attributes
    ----------
    data : pandas DataFrame
        Annotation DataFranme.

    Methods
    -------
    check_integrity(verbose=False, time_duplicates_only=False)
        Check integrity of Annotation object.
    from_raven(files, class_header='Sound type',subclass_header=None,verbose=False)
        Import annotation data from 1 or several Raven files.
    to_raven(outdir, single_file=False)
        Write annotation data to one or several Raven files.
    from_pamlab(files, verbose=False)
        Import annotation data from 1 or several PAMlab files.
    to_pamlab(outdir, single_file=False)
        Write annotation data to one or several Raven files.
    from_parquet(file)
        Import annotation data from a Parquet file.
    to_parquet(file)
        Write annotation data to a Parquet file.
    insert_values(**kwargs)
        Manually insert values for given Annotation fields.
    get_labels_class()
        Return all unique class labels.
    get_labels_subclass()
        Return all unique subclass labels.
    get_fields()
        Return list with all annotations fields.
    __add__()
        Concatenate data from annotation objects uisng the + sign.
    __len__()
        Return number of annotations.
    """

    def __init__(self):
        """
        Initialize Annotation object.

        Sets all the annotation fields.:
            -'uuid': UUID,
                Unique identifier code
            -'from_detector': bool,
                True if data comes from an automatic process.
            -'software_name': str,
                Software name. Can be Raven or PAMlab for manual analysis.
            -'software_version': str,
                Version of the software used to create the annotations.
            -'operator_name': str,
                Name of the person responsible for the creation of the
                annotations.
            -'UTC_offset': float,
                Offset hours to UTC.
            -'entry_date': datetime,
                Date when the annotation was created.
            -'audio_channel': int,
                Channel number.
            -'audio_file_name': str,
                Name of the audio file.
            -'audio_file_dir': str,
                Directory where the audio file is.
            -'audio_file_extension': str,
                Extension of teh audio file.
            -'audio_file_start_date': datetime,
                Date of the audio file start time.
            -'audio_sampling_frequency': int,
                Sampling frequecy of the audio data.
            -'audio_bit_depth': int,
                Bit depth of the audio data.
            -'mooring_platform_name': str,
                Name of the moorig platform (e.g. 'glider','Base plate').
            -'recorder_type': str,
                Name of the recorder type (e.g., 'AMAR'), 'SoundTrap'.
            -'recorder_SN': str,
                Serial number of the recorder.
            -'hydrophone_model': str,
                Model of the hydrophone.
            -'hydrophone_SN': str,
                Serial number of the hydrophone.
            -'hydrophone_depth': float,
                Depth of the hydrophone in meters.
            -'location_name': str,
                Name of the deploymnet location.
            -'location_lat': float,
                latitude of the deployment location in decimal degrees.
            -'location_lon': float,
                longitude of the deployment location in decimal degrees.
            'location_water_depth': float,
                Water depth at the deployment location in meters.
            -'frequency_min': float,
                Minimum frequency of the annotaion in Hz.
            -'frequency_max': float,
                Maximum frequency of the annotaion in Hz.
            -'time_min_offset': float,
                Start time of the annotaion, in seconds relative to the
                begining of the audio file.
            -'time_max_offset': float,
                Stop time of the annotaion, in seconds relative to the
                begining of the audio file.
            -'time_min_date': datetime,
                Date of the annotation start time.
            -'time_max_date': datetime,
                Date of the annotation stop time.
            -'duration': float,
                Duration of the annotation in seconds.
            -'label_class': str,
                label of the annotation class (e.g. 'fish').
            -'label_subclass': str,
                label of the annotation subclass (e.g. 'grunt')
            'confidence': float,
                Confidence of the classification.

        Returns
        -------
        Empty Annotation object.

        """
        self.data = pd.DataFrame({
            'uuid': [],
            'from_detector': [],  # True, False
            'software_name': [],
            'software_version': [],
            'operator_name': [],
            'UTC_offset': [],
            'entry_date': [],
            'audio_channel': [],
            'audio_file_name': [],
            'audio_file_dir': [],
            'audio_file_extension': [],
            'audio_file_start_date': [],
            'audio_sampling_frequency': [],
            'audio_bit_depth': [],
            'mooring_platform_name': [],
            'recorder_type': [],
            'recorder_SN': [],
            'hydrophone_model': [],
            'hydrophone_SN': [],
            'hydrophone_depth': [],
            'location_name': [],
            'location_lat': [],
            'location_lon': [],
            'location_water_depth': [],
            'frequency_min': [],
            'frequency_max': [],
            'time_min_offset': [],
            'time_max_offset': [],
            'time_min_date': [],
            'time_max_date': [],
            'duration': [],
            'label_class': [],
            'label_subclass': [],
            'confidence': []
            })

    def check_integrity(self, verbose=False, ignore_frequency_duplicates=False):
        """
        Check integrity of Annotation object.

        Tasks performed:
            1- Check that start time < stop time
            2- Check that min frequency < max frequency
            3- Remove duplicate entries based on time and frequency, filename,
               labels and filenames

        Parameters
        ----------
        verbose : bool, optional
            Print summary of the duplicate entries deleted.
            The default is False.
        ignore_frequency_duplicates : bool, optional
            If set to True, doesn't consider frequency values when deleting
            duplicates. It is useful when data are imported from Raven.
            The default is False.

        Raises
        ------
        ValueError
            If annotations have a start time > stop time
            If annotations have a min frequency > max frequency

        Returns
        -------
        None.

        """
        # Drop all duplicates
        count_start = len(self.data)
        if ignore_frequency_duplicates:  # doesn't use frequency boundaries
            self.data = self.data.drop_duplicates(
                subset=['time_min_offset',
                        'time_max_offset',
                        'label_class',
                        'label_subclass',
                        'audio_file_name',
                        ], keep="first",).reset_index(drop=True)
        else:  # remove annot with exact same time AND frequency boundaries
            self.data = self.data.drop_duplicates(
                subset=['time_min_offset',
                        'time_max_offset',
                        'frequency_min',
                        'frequency_max',
                        'label_class',
                        'label_subclass',
                        'audio_file_name',
                        ], keep="first",).reset_index(drop=True)
        count_stop = len(self.data)
        if verbose:
            print('Duplicate entries removed:', str(count_start-count_stop))
        # Check that start and stop times are coherent (i.e. t2 > t1)
        time_check = self.data.index[
            self.data['time_max_offset'] <
            self.data['time_min_offset']].tolist()
        if len(time_check) > 0:
            raise ValueError(
                'Incoherent annotation times (time_min > time_max). \
                 Problematic annotations:' + str(time_check))
        # Check that min and max frequencies are coherent (i.e. fmin < fmax)
        freq_check = self.data.index[
            self.data['frequency_max'] < self.data['frequency_min']].tolist()
        if len(freq_check) > 0:
            raise ValueError(
                'Incoherent annotation frequencies (frequency_min > \
                frequency_max). Problematic annotations:' + str(freq_check))
        if verbose:
            print('Integrity test succesfull')

    def from_raven(self, files, class_header='Sound type', subclass_header=None, verbose=False):
        """
        Import data from 1 or several Raven files.

        Load annotation tables from .txt files gnereated by the software Raven.

        Parameters
        ----------
        files : str, list
            Path of the txt file to import. Can be a str if importing a single
            file. Needs to be a list if importing multiple files.
        class_header : str, optional
            Name of the header in the Raven file corresponding to the class
            name. The default is 'Sound type'.
        subclass_header : str, optional
            Name of the header in the Raven file corresponding to the subclass
            name. The default is None.
        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
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
            self.data['audio_file_start_date'] + pd.to_timedelta(
                self.data['time_min_offset'], unit='s'))
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
        self.data['uuid'] = self.data.apply(lambda _: str(uuid.uuid4()), axis=1)
        self.data['duration'] = self.data['time_max_offset'] - self.data['time_min_offset']
        self.check_integrity(verbose=verbose, ignore_frequency_duplicates=True)

    def to_raven(self, outdir, single_file=False):
        """
        Write data to 1 or several Raven files.

        Write annotations as .txt files readable by the software Raven. Output
        files can be written in a single txt file or in several txt files (one
        per audio recording). Output file names are automatically generated
        based on the audio file's name.

        Parameters
        ----------
        outdir : str
            Path of the output directory where the Raven files are written.
        single_file : bool, optional
            If set to True, writes a single output file with all annotations.
            The default is False.

        Returns
        -------
        None.

        """
        if single_file:
            annots = [self.data]
        else:
            annots = [pd.DataFrame(y) for x, y in self.data.groupby(
                'audio_file_name', as_index=False)]
        for annot in annots:
            annot.reset_index(inplace=True, drop=True)
            cols = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                    'End Time (s)', 'Delta Time (s)', 'Low Freq (Hz)',
                    'High Freq (Hz)', 'Begin Path', 'File Offset (s)',
                    'Begin File', 'Class', 'Sound type', 'Software',
                    'Confidence']
            outdf = pd.DataFrame({'Selection': 0, 'View': 0, 'Channel': 0,
                                  'Begin Time (s)': 0, 'End Time (s)': 0,
                                  'Delta Time (s)': 0, 'Low Freq (Hz)': 0,
                                  'High Freq (Hz)': 0, 'Begin Path': 0,
                                  'File Offset (s)': 0, 'Begin File': 0,
                                  'Class': 0, 'Sound type': 0, 'Software': 0,
                                  'Confidence': 0},
                                 index=list(range(annot.shape[0])))
            outdf['Selection'] = range(1, annot.shape[0]+1)
            outdf['View'] = 'Spectrogram 1'
            outdf['Channel'] = annot['audio_channel']
            outdf['Begin Time (s)'] = annot['time_min_offset']
            outdf['End Time (s)'] = annot['time_max_offset']
            outdf['Delta Time (s)'] = annot['duration']
            outdf['Low Freq (Hz)'] = annot['frequency_min']
            outdf['High Freq (Hz)'] = annot['frequency_max']
            outdf['File Offset (s)'] = annot['time_min_offset']
            outdf['Class'] = annot['label_class']
            outdf['Sound type'] = annot['label_subclass']
            outdf['Software'] = annot['software_name']
            outdf['Confidence'] = annot['confidence']
            outdf['Begin Path'] = [os.path.join(x, y) + z
                                  for x, y, z in zip(annot['audio_file_dir'],
                                                     annot['audio_file_name'],
                                                     annot['audio_file_extension'])]
            outdf['Begin File'] = [x + y
                                  for x, y in zip(annot['audio_file_name'],
                                                  annot['audio_file_extension'])]
            outdf = outdf.fillna(0)
            outdf.to_csv(os.path.join(
                outdir, str(annot['audio_file_name'].iloc[0]))
                + str(annot['audio_file_extension'].iloc[0])
                + '.chan' + str(annot['audio_channel'].iloc[0])
                + '.Table.1.selections.txt',
                sep='\t',
                encoding='utf-8',
                header=True,
                columns=cols,
                index=False)

    def from_pamlab(self, files, verbose=False):
        """
        Import data from 1 or several Raven files.

        Load annotation data from .log files gnereated by the software PAMlab.

        Parameters
        ----------
        files : str, list
            Path of the txt file to import. Can be a str if importing a single
            file. Needs to be a list if importing multiple files.
        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
        data = Annotation._import_files(files)
        files_timestamp = core.tools.filename_to_datetime(
            data['Soundfile'].tolist())
        self.data['audio_file_start_date'] = files_timestamp
        self.data['operator_name'] = data['Operator']
        self.data['entry_date'] = pd.to_datetime(
            data['Annotation date and time (local)'],
            format='%Y-%m-%d %H:%M:%S.%f')
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
            self.data['audio_file_start_date']
            + pd.to_timedelta(self.data['time_min_offset'], unit='s'))
        self.data['time_max_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] +
            pd.to_timedelta(self.data['time_max_offset'], unit='s'))
        self.data['frequency_min'] = data['Bottom freq (Hz)']
        self.data['frequency_max'] = data['Top freq (Hz)']
        self.data['label_class'] = data['Species']
        self.data['label_subclass'] = data['Call type']
        self.data['from_detector'] = False
        self.data['software_name'] = 'pamlab'
        self.data['uuid'] = self.data.apply(lambda _: str(uuid.uuid4()), axis=1)
        self.data['duration'] = self.data['time_max_offset'] - self.data['time_min_offset']
        self.check_integrity(verbose=verbose)

    def to_pamlab(self, outdir, single_file=False):
        """
        Write data to 1 or several PAMlab files.

        Write annotations as .log files readable by the software PAMlab. Output
        files can be written in a single txt file or in several txt files (one
        per audio recording). Output file names are automatically generated
        based on the audio file's name and the name format required by PAMlab.

        Parameters
        ----------
        outdir : str
            Path of the output directory where the Raven files are written.
        single_file : bool, optional
            If set to True, writes a single output file with all annotations.
            The default is False.

        Returns
        -------
        None.

        """
        if single_file:
            annots = [self.data]
        else:
            annots = [pd.DataFrame(y)
                      for x, y in self.data.groupby(
                              'audio_file_name', as_index=False)]
        for annot in annots:
            annot.reset_index(inplace=True, drop=True)
            cols = ['fieldkey:', 'Soundfile', 'Channel', 'Sampling freq (Hz)',
                    'Latitude (deg)', 'Longitude (deg)', 'Recorder ID',
                    'Recorder depth', 'Start date and time (UTC)',
                    'Annotation date and time (local)', 'Recorder type',
                    'Deployment', 'Station', 'Operator', 'Left time (sec)',
                    'Right time (sec)', 'Top freq (Hz)', 'Bottom freq (Hz)',
                    'Species', 'Call type', 'rms SPL', 'SEL', '', '']
            outdf = pd.DataFrame({'fieldkey:': 0, 'Soundfile': 0, 'Channel': 0,
                                  'Sampling freq (Hz)': 0, 'Latitude (deg)': 0,
                                  'Longitude (deg)': 0, 'Recorder ID': 0,
                                  'Recorder depth': 0,
                                  'Start date and time (UTC)': 0,
                                  'Annotation date and time (local)': 0,
                                  'Recorder type': 0, 'Deployment': 0,
                                  'Station': 0, 'Operator': 0,
                                  'Left time (sec)': 0, 'Right time (sec)': 0,
                                  'Top freq (Hz)': 0, 'Bottom freq (Hz)': 0,
                                  'Species': '', 'Call type': '', 'rms SPL': 0,
                                  'SEL': 0, '': '', '': ''},
                                 index=list(range(annot.shape[0])))
            outdf['fieldkey:'] = 'an:'
            outdf['Species'] = annot['label_class']
            outdf['Call type'] = annot['label_subclass']
            outdf['Left time (sec)'] = annot['time_min_offset']
            outdf['Right time (sec)'] = annot['time_max_offset']
            outdf['Top freq (Hz)'] = annot['frequency_max']
            outdf['Bottom freq (Hz)'] = annot['frequency_min']
            outdf['rms SPL'] = annot['confidence']
            outdf['Operator'] = annot['operator_name']
            outdf['Channel'] = annot['audio_channel']
            outdf['Annotation date and time (local)'] = annot['entry_date']
            outdf['Sampling freq (Hz)'] = annot['audio_sampling_frequency']
            outdf['Recorder type'] = annot['recorder_type']
            outdf['Recorder ID'] = annot['recorder_SN']
            outdf['Recorder depth'] = annot['hydrophone_depth']
            outdf['Station'] = annot['location_name']
            outdf['Latitude (deg)'] = annot['location_lat']
            outdf['Longitude (deg)'] = annot['location_lon']
            outdf['Soundfile'] = [os.path.join(x,y) + z
                                  for x, y, z in zip(annot['audio_file_dir'],
                                                     annot['audio_file_name'],
                                                     annot['audio_file_extension'])]
            outdf = outdf.fillna(0)
            outdf.to_csv(os.path.join(outdir, str(annot['audio_file_name'].iloc[0]))
                         + str(annot['audio_file_extension'].iloc[0])
                         + ' annotations.log',
                         sep='\t',
                         encoding='utf-8',
                         header=True,
                         columns=cols,
                         index=False)

    def from_parquet(self, file):
        """
        Import data to a Parquet file.

        Load annotations from a .parquet file. This format allows for fast and
        efficient data storage and access.

        Parameters
        ----------
        file : str
            Path of the output directory where the Raven files are written.

        Returns
        -------
        None.

        """
        self.data = pd.read_parquet(file)

    def to_parquet(self, file):
        """
        Write data to a Parquet file.

        Write annotations as .parquet file. This format allows for fast and
        efficient data storage and access.

        Parameters
        ----------
        file : str
            Path of the output directory where the Raven files are written.

        Returns
        -------
        None.

        """
        self.data.to_parquet(file,
                             coerce_timestamps='ms',
                             allow_truncated_timestamps=True)

    def insert_values(self, **kwargs):
        """
        nsert constant values for given Annotation fields.

        Fill in entire columns of the annotation dataframe with constant
        values. It is usefull for adding project related informations that may
        not be included in data imported from Raven or PAMlab files (e.g.,
        'location_lat', 'location_lon'). Values can be inserted for several
        annattions fields at a time by setting several keywords. This should
        only be used for filling in static values (i.e., not for variable
        values such as time/frequency boundaries of the annotations). Keywords
        must have the exact same name as the annotation field (see method
        .get_fields). For example: (location_lat=48.6, recorder_type='AMAR')

        Parameters
        ----------
        **kwargs : annotation filed name
            Keyword and value of the annotation field to fill in. Keywords must
            have the exact same name as the annotation field.

        Raises
        ------
        ValueError
            If keyword doesn't match any annotation field name.

        Returns
        -------
        None.

        """
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key] = value
            else:
                raise ValueError('The annotation object has no field: '
                                 + str(key))

    def get_labels_class(self):
        """
        Get all the unique class labels of the annotations.

        Returns
        -------
        classes : list
            List of unique class labels.

        """
        if len(self.data) > 0:
            classes = list(self.data['label_class'].unique())
        else:
            classes = []
        return classes

    def get_labels_subclass(self):
        """
        Get all the unique subclass labels of the annotations.

        Returns
        -------
        classes : list
            List of unique subclass labels.

        """
        if len(self.data) > 0:
            subclasses = list(self.data['label_subclass'].unique())
        else:
            subclasses = []
        return subclasses

    def get_fields(self):
        """
        Get all the annotations fields.

        Returns
        -------
        classes : list
            List of annotation fields.

        """
        return list(self.data.columns)

    @staticmethod
    @core.decorators.listinput
    def _import_files(files):
        """Import one or several text files with header to a Panda datafrane."""
        assert type(files) in (str, list), "Input must be of type str (single \
            file) or list (multiple files)"
        # Import all files to a dataframe
        for idx, file in enumerate(files):
            # Extract header first due to formating issues in PAMlab files
            header = pd.read_csv(file,
                                 delimiter='\t',
                                 header=None,
                                 nrows=1)
            headerLength = header.shape[1]
            # Get all data and only keep values correpsonding to header labels
            tmp = pd.read_csv(file,
                              delimiter='\t',
                              header=None,
                              skiprows=1,
                              na_values=None)
            tmp = tmp.iloc[:, 0:headerLength]
            # Put header back
            tmp = tmp.set_axis(list(header.values[0]), axis=1, inplace=False)
            if idx == 0:
                data = tmp
            else:
                data = pd.concat([data, tmp], ignore_index=True, sort=False)
        return data

    def __add__(self, other):
        """Concatenate data from several annotation objects."""
        assert type(other) is core.annotation.Annotation, "Object type not \
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
