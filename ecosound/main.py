from core.annotation import Annotation


PAMlab_files = []
Raven_files = []
Raven_files.append(r".\resources\AMAR173.4.20190916T061248Z.Table.1.selections.txt")
annot1 = Annotation()
annot1.from_raven(Raven_files, verbose=False)
print(len(annot1))

Raven_files = []
Raven_files.append(r".\resources\67674121.181018013806.Table.1.selections.txt")
annot2 = Annotation()
annot2.from_raven(Raven_files, verbose=False)
print(len(annot2))

PAMlab_files = []
PAMlab_files.append(r".\resources\JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav annotations.log")
annot3 = Annotation()
annot3.from_pamlab(PAMlab_files, verbose=False)
print(len(annot3))

annot = annot1 + annot2 + annot3
print(len(annot))

print(annot.get_fields())
annot.insert_values(operator_name='Xavier Mouy')
