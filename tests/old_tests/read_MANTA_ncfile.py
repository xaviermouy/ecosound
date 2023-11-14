from ecosound.core.annotation import Annotation

detec_file = r'C:\Users\xavier.mouy\Desktop\detections.nc'

detec = Annotation()
detec.from_netcdf(detec_file)
