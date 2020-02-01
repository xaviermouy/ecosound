from core.annotation import Annotation
from core.tools import filename_to_datetime

# PAMlab_files = []
# PAMlab_files.append(r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log")
# AMlab_files.append(r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log")
# PAMlab_files.append(r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations_2.log")
# PAMlab_files = r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log"
# annot= Annotation()
# annot.from_pamlab(PAMlab_files)


#PAMlab_files = []
Raven_files = []
#Raven_files.append(r".\resources\67674121.181018013806.Table.1.selections.txt")
Raven_files.append(r".\resources\AMAR173.4.20190916T061248Z.Table.1.selections.txt")
annot= Annotation()
annot.from_raven(Raven_files,verbose=True)

