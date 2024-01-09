from ecosound.core.tools import filename_to_datetime

file_1 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\ACOUSTIC_DATA\BOTTOM_MOUNTED\NEFSC_SBNMS\NEFSC_SBNMS_202103_SB01\67403784_48kHz\SanctSound_SB01_15_67403784_20210324_223239.wav'
file_2 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\ACOUSTIC_DATA\BOTTOM_MOUNTED\NEFSC_SBNMS\NEFSC_SBNMS_200606\NEFSC_SBNMS_200606_EST\20060629\SNMS_20060629_023000.aif'
file_3 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\ACOUSTIC_DATA\BOTTOM_MOUNTED\JASCO_DE\JASCO_DE_201006\MMS_Dep1_Delaware-e1d282a3.e1d282a3.Chan_1-24bps.1277236867.2010-06-22-20-01-07.wav'
file_4 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\ACOUSTIC_DATA\BOTTOM_MOUNTED\NEFSC_MA-RI\NEFSC_MA-RI_202202_NS01\NEFSC_MA-RI_202202_NS01_ST\6075_64kHz_UTC\6075.220210084905.wav'
file_5 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\PAM_20171108_094819_000.wav'
file_6 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\WAT_NC_03_170716_180000_df100.x'
file_7 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\SanctSound_SB01_01_1678032935_181112190000.wav'
file_8 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\Brunswick04_002K_M06_multi_UTCm5_20170528_110000.aif'
file_9 = r'\\stellwagen.nefsc.noaa.gov\stellwagen\NRS08-140523-001620.wav'


print(filename_to_datetime(file_1))
print(filename_to_datetime(file_2))
print(filename_to_datetime(file_3))
print(filename_to_datetime(file_4))
print(filename_to_datetime(file_5))
print(filename_to_datetime(file_6))
print(filename_to_datetime(file_7))
print(filename_to_datetime(file_8))
print(filename_to_datetime(file_9))
