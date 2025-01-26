from ecosound.core.annotation import Annotation
from ecosound.core.tools import list_files, filename_to_datetime
import datetime
import os

in_dir = r'C:\Users\xavier.mouy\Desktop\fichiers_nc'

files = list_files(in_dir,'.nc')
for file in files:
    annot = Annotation()
    annot.from_netcdf(file)
    for n in range(0,len(annot)):
        annot.data['time_min_date'].iloc[n] = filename_to_datetime(file)[0] + datetime.timedelta(seconds=annot.data.iloc[n]['time_min_offset'])
        annot.data['time_max_date'].iloc[n] = filename_to_datetime(file)[0] + datetime.timedelta(seconds=annot.data.iloc[n]['time_max_offset'])
    annot.to_netcdf(file)

print('done')


