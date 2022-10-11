# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:34:38 2022

@author: xavier.mouy
"""
import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.
from ecosound.core.annotation import Annotation
import ecosound


annot_file = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\Annotations_dataset_FRA-NEFSC-CARIBBEAN-201612-MTQ annotations.nc'
detec_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ\detections.sqlite'
thresholds = np.arange(0.5,1.01,0.01)
class_pos ='MW'

eval_start_datetime = 
eval_end_datetime = 


# load ground truth data
annot = Annotation()
annot.from_netcdf(annot_file, verbose=True)

# load detections
detec = Annotation()
detec.from_sqlite(detec_file, verbose=True)


# annot = pd.read_csv(annot_file)

# # initialize TP, FP, TP
# TP = np.zeros((len(annot),len(thresholds))) # True positives (annot x threshold)
# FP = np.zeros((len(annot),len(thresholds))) # False positives (annot x threshold)
# FN = np.zeros((len(annot),len(thresholds))) # False negatives (annot x threshold)
# TN = np.zeros((len(annot),len(thresholds))) # True negatives (annot x threshold)

# # go through each annotation
# for an_idx,  an in annot.iterrows():
#     print(an_idx, an['audio_file_name'])
    
#     # Annotation label, and start/stop stimes
#     an_label = an['label']
#     an_t1 = an['time_min_offset']
#     an_t2 = an['time_max_offset']
    
#     #if an_label == class_pos:
#     #    print('stop here')
    
#     # load detection file
#     try:
#         detec = pd.read_csv(os.path.join(detec_dir, an['audio_file_name']+'.wav.chan1.Table.1.selections.txt'),sep='\t')
#     except:
#         detec = None # no detections at all
    
#     # go through thresholds
#     for th_idx, th in enumerate(thresholds):
#         #print(th)
#         # only keeps detectio above current threshold
#         if detec is not None:
#             detec_th = detec[detec['Confidence']>=th]
#             if len(detec_th) == 0:
#                 detec_th = None
#         else:
#             detec_th = None
        
#         # find detections overlapping with annotation
#         is_detec = False
#         if detec_th is not None: # if there are any detections left at this threshold            
#             for _ , tmp in detec_th.iterrows():
#                 det_t1 = tmp['Begin Time (s)']
#                 det_t2 = tmp['End Time (s)']
                
#                 is_overlap = (((det_t1 <= an_t1) & (det_t2 >= an_t2)) |             # 1- annot inside detec
#                          ((det_t1 >= an_t1) & (det_t2 <= an_t2)) |                  # 2- detec inside annot
#                          ((det_t1 < an_t1) & (det_t2 < an_t2) & (det_t2 > an_t1)) | # 3- only the end of the detec overlaps with annot
#                          ((det_t1 > an_t1) & (det_t1 < an_t2) & (det_t2 > an_t2)))  # 4- only the begining of the detec overlaps with annot
                          
#                 if is_overlap == True:
#                     is_detec = True
#                     break
#         # count TP, FP, FN
#         if (an_label == class_pos) &  (is_detec == True): # TP
#             TP[an_idx,th_idx] = 1 
#         if (an_label != class_pos) &  (is_detec == False): # TN
#             TN[an_idx,th_idx] = 1 
#         if (an_label == class_pos) &  (is_detec == False): # FN
#             FN[an_idx,th_idx] = 1 
#         if (an_label != class_pos) &  (is_detec == True): # FP
#             FP[an_idx,th_idx] = 1 
        
# # sum up             
# TP_count = sum(TP)
# TN_count = sum(TN)
# FN_count = sum(FN)
# FP_count = sum(FP)

# # calculate metrics for each trheshold
# P=np.zeros(len(thresholds)) 
# R=np.zeros(len(thresholds)) 
# F=np.zeros(len(thresholds)) 
# for th_idx in range(0,len(thresholds)):
#     P[th_idx] = TP_count[th_idx] / (TP_count[th_idx] + FP_count[th_idx])
#     R[th_idx] = TP_count[th_idx] / (TP_count[th_idx] + FN_count[th_idx])
#     F[th_idx] = (2*P[th_idx]*R[th_idx])/(P[th_idx]+R[th_idx])


# # save
# df = pd.DataFrame({'Thresholds':thresholds, 'P': P, 'R': R, 'F':F})
# df.to_csv(os.path.join(detec_dir,'performance.csv'))
# df2 = pd.DataFrame(TP)
# df2 = pd.concat([annot['audio_file_name'], df2], ignore_index=True,axis=1)
# df2.to_csv(os.path.join(detec_dir,'TP.csv'))
# df3 = pd.DataFrame(FP)
# df3 = pd.concat([annot['audio_file_name'], df3], ignore_index=True,axis=1)
# df3.to_csv(os.path.join(detec_dir,'FP.csv'))
# df4 = pd.DataFrame(FN)
# df4 = pd.concat([annot['audio_file_name'], df4], ignore_index=True,axis=1)
# df4.to_csv(os.path.join(detec_dir,'FN.csv'))

# ## Graphs          
# plt.plot(P,R,'k')
# plt.xlabel('Precision')
# plt.ylabel('Recall')
# plt.grid()
# plt.xticks(np.arange(0, 1+0.02, 0.02))
# plt.yticks(np.arange(0, 1+0.02, 0.02))
# plt.ylim([0.8,1])
# plt.xlim([0.8,1])
# plt.savefig(os.path.join(detec_dir,'PR_curve.png'))

# plt.figure() 
# plt.plot(thresholds,R,':k', label='Recall')
# plt.plot(thresholds,P,'--k', label='Precision')
# plt.plot(thresholds,F,'k', label='F-score')
# plt.legend()
# plt.xlabel('Threshold')
# plt.ylabel('Performance')
# plt.grid()
# plt.xticks(np.arange(0, 1+0.05, 0.05))
# plt.yticks(np.arange(0, 1+0.02, 0.02))
# plt.ylim([0.8,1])
# plt.xlim([thresholds[0],thresholds[-1]])
# plt.savefig(os.path.join(detec_dir,'PRF_curves.png'))


        
        
    
    