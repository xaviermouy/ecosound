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
thresholds = np.arange(0, 1, 0.01)
# thresholds = [0.9]

# Name of the class to test
target_class = "MW"

# List of files to use
files_to_use = "detec"  # 'detec', 'annot', 'both', list

# restrictions on min and max dates:
date_min = None  # date string or None
date_max = None  # date string or None

# Beta parameter for calculating the F-score
F_beta = 1

# Matching criteria
freq_ovp = True
dur_factor_max = None
dur_factor_min = None
ovlp_ratio_min = None
remove_duplicates = False
inherit_metadata = False
filter_deploymentID = False
do_plot = False

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
print("Calculating performance...")
PRF.count(
    annot=annot,
    detec=detec,
    out_dir=out_dir,
    target_class=target_class,
    files_to_use=files_to_use,
    date_min=date_min,
    date_max=date_max,
    thresholds=thresholds,
    F_beta=F_beta,
    freq_ovp=freq_ovp,
    dur_factor_max=dur_factor_max,
    dur_factor_min=dur_factor_min,
    ovlp_ratio_min=ovlp_ratio_min,
    remove_duplicates=remove_duplicates,
    inherit_metadata=inherit_metadata,
    filter_deploymentID=filter_deploymentID,
    do_plot=do_plot,
)
