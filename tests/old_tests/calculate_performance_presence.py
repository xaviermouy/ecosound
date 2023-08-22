# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:16:31 2022

@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
from ecosound.evaluation.prf import PRF
import numpy as np

###############################################################################
##  input parameters ##########################################################
###############################################################################

# # Annotation file:
# annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations_withSNR.nc"
# # Detection folder or detection sqllite file
# detec_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\results\UK-SAMS-N1\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm_models-5s_bs128_ep50\detections.sqlite"
# # output folder
# out_dir = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\results\UK-SAMS-N1\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm_models-5s_bs128_ep50\performance_results"

# Annotation file:
annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations_withSNR.nc"
# Detection folder or detection sqllite file
detec_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\results\UK-SAMS-N1\spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm_models-5s_bs128_ep50\detections.sqlite"
# output folder
out_dir = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\results\UK-SAMS-N1\spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm_models-5s_bs128_ep50\performance_results"


# Detection threshold to test
thresholds = np.arange(0, 1.05, 0.05)
# thresholds = [0.9]

# Name of the class to test
target_class = "MW"

# List of files to use
files_to_use = "detec"  # 'detec', 'annot', 'both', list

# restrictions on min and max dates:
date_min = None  # date string or None
date_max = None  # date string or None ("2017-02-23 18:16:30")

# Integration tine to use and minimum number of detections needed to consider a detection
integration_time = "1D"  # can be '1Min', '1H', '2H',... '1D'
min_detec_nb = 1

# Beta parameter for calculating the F-score
F_beta = 1

###############################################################################
###############################################################################

# load ground truth data
print(" ")
print("Loading manual annotations...")
annot = Annotation()
annot.from_netcdf(annot_file)
print("Annotation labels:")
print(annot.get_labels_class())

# load destections
print(" ")
print("Loading automatic detections...")
detec = Annotation()
detec.from_sqlite(detec_file)

# calculate performance
print(" ")
print("CCalculating performance...")
PRF.presence(
    annot=annot,
    detec=detec,
    out_dir=out_dir,
    target_class=target_class,
    files_to_use=files_to_use,
    date_min=date_min,
    date_max=date_max,
    thresholds=thresholds,
    F_beta=F_beta,
    integration_time=integration_time,
    min_detec_nb=min_detec_nb,
)
