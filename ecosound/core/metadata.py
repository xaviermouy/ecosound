# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:56:38 2020

@author: xavier.mouy
"""

import pandas as pd
import os

class DeploymentInfo():

    def __init__(self):
        self.data =[];
    
    def write_template(self, filepath):
        """
        Create a blank deployment file.
    
        Parameters
        ----------
        filepath : str
            path and name of the deployment csv file to create.
    
        Returns
        -------
        None. Write a blank csv deployment file that users can fill in.
    
        """
        if os.path.isfile(filepath):
            raise ValueError('File already exists.')
            
        metadata = pd.DataFrame({
            'UTC_offset': [],
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
            'deployment_ID': [],
            'deployment_date':[],
            'recovery_date':[],
            })
        metadata.to_csv(filepath,
                        sep=',',
                        encoding='utf-8',
                        header=True,
                        index=False,
                        )
    
    def read(self, filepath):
        df = pd.read_csv(filepath,
                         delimiter=',',
                         #header=None,
                         skiprows=0,
                         na_values=None,
                         )
        self.data = df
        return df
            
