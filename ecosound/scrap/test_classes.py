# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
import pandas as pd

# class TimeFrequencyBox:
#     """Defines a time-frequency box object."""
    
#     def __init__(self,t_min,t_max,f_min,f_max):
#         assert t_max > t_min, 't_max must be > t_min'
#         assert f_max > f_min, 'f_max must be > f_min'
#         self.t_min = t_min
#         self.t_max = t_max
#         self.f_min = f_min
#         self.f_max = f_max
#     def talk(self):
#         """Says Yeah."""
#         print('Yeah!')
#     def __repr__(self):
#         return 'I am a TF box object'

# class DetectionBox(TimeFrequencyBox):
#     """Defines a detetion object."""
    
#     def __init__(self,t_min,t_max,f_min,f_max):
#         super().__init__(t_min,t_max,f_min,f_max)
#         self.detectorname=[]
#     def __repr__(self):
#         return 'I am an bobject X'

# class DetectionContour(DetectionBox):
#     """Defines a detection contour."""
    
#     def __init__(self):
#         super().__init__()
#         self.contour=[]

class Annotation():
    """Defines an annotation object."""
   
    def __init__(self):
        self.dataframe = pd.DataFrame({
            'UID':[],
            'operator_type':[],
            'operator_ID':[],
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

    def read_raven(self, files):
        """import from 1 or several Raven files."""
    def to_raven(self, file):
        """write to a Raven files."""
    
    def read_pamlab(self, files):
        """import from 1 or several PAMLab files."""
        self. df = pd.read_csv(files, sep='\t', encoding='utf-8', header=True, index=False)
        return self
    
    def to_pamlab(self, file):
        """write to a PAMLab file."""        

    def read_pytable(self, conn):
        """import from pytable file."""
        
    def print():
        print('dsdsd')
            
    def __repr__(self):
        return 'Annotation object'

PAMlab_files = []
PAMlab_files.append(r"C:\Users\xavier.mouy\Documents\Workspace\GitHub\FishSoundDetector\data\PAMlab_annot1.log")
PAMlab_files.append(r"C:\Users\xavier.mouy\Documents\Workspace\GitHub\FishSoundDetector\data\PAMlab_annot2.log")
annot= Annotation()
#an2 = annot.read_pamlab(PAMlab_files[1])

data= pd.DataFrame({
            'UID':[],
            'operator_type':[],
            'operator_ID':[],
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

files = PAMlab_files[0]

if type(files) is str:
    files = [files]
#assert list of str...

df = pd.concat(map(lambda file: pd.read_csv(file, sep='\t', encoding='utf-8'), files))

#df.rename(columns={"A": "a", "B": "c"})



#lambda p: myFunc(p, additionalArgument)
    
#     dfs.append(pd.read_csv(PAMlab_file))
#     temp = pd.read_csv(PAMlab_files, sep='\t', encoding='utf-8')
# print(len(temp))