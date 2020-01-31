from core.annotation import Annotation
from core.tools import filename_to_datetime

PAMlab_files = []
PAMlab_files.append(r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log")
PAMlab_files.append(r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log")
#PAMlab_files.append(r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations_2.log")
#PAMlab_files = r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log"

annot= Annotation()
annot.from_pamlab(PAMlab_files)


