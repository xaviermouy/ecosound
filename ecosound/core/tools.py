# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:21:40 2020

@author: xavier.mouy
"""
import json
import re
from datetime import datetime
import core.decorators


def read_json(file):
    """Load JSON file as dict."""
    with open(file, "r") as read_file:
        data = json.load(read_file)
    return data

@core.decorators.listinput
def filename_to_datetime(files):
    """Extract date from a list of str of filenames."""
    patterns = read_json(r'./core/timestamp_formats.json')
    regex_string ='|'.join([pattern['string_pattern'] for pattern in patterns])
    time_formats =[pattern['time_format'] for pattern in patterns]
    timestamps =[None]*len(files)
    p = re.compile(regex_string)
    for idx, file in enumerate(files):
        datestr = p.search(file)
        for time_format in time_formats:        
            ok_flag = False
            try:
                timestamps[idx] = datetime.strptime(datestr[0], time_format)
                ok_flag = True
            except:
                ok_flag = False
            if ok_flag is True:
                break
        if ok_flag is False:
            raise ValueError('Time format not recognized:'+ file)
    return timestamps


