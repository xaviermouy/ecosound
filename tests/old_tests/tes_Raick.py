
from ecosound.core.annotation import Annotation
from ecosound.core.tools import filename_to_datetime
#root_dir = r'C:\Users\xavier.mouy\Desktop\raick'
#annot = Annotation()
#annot.from_raven(root_dir)

# print('d')
#
# file =r'C:\Users\xavie\Desktop\WHOI_fish\data_training_and_evaluation\20220711T174119.wav'
# file2 = r"C:\Users\xavie\Desktop\WHOI_fish\data_training_and_evaluation_1bigfile\20220711T170024.wav"
#
# dep_file =r'C:\Users\xavier.mouy\Desktop\raick\deployment_info_test2.csv'
# date = filename_to_datetime(file)
#
#
# from ecosound.core.metadata import DeploymentInfo
# dep_info = DeploymentInfo()
# print()



annot_file = r'C:\Users\xavier.mouy\Desktop\raick\New folder'
annot = Annotation()
annot.from_raven(annot_file, class_header='Annotation')
print('s')
