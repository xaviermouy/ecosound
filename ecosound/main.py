from bioacoustic_toolkit.core.annotation import Annotation


PAMlab_files = []
#PAMlab_files.append(r"C:\Users\xavier.mouy\Documents\Workspace\GitHub\FishSoundDetector\data\671404070.190722164336.wav chan0 annotations.log")
PAMlab_files.append(r".\ressources\AMAR173.4.20190920T161248Z.wav annotations.log")

annot= Annotation()
annot.read_pamlab(PAMlab_files)

# #df.rename(columns={"A": "a", "B": "c"})

# #df1 = pd.concat(map(lambda PAMlab_files: pd.read_csv(PAMlab_files, delimiter='\t'), PAMlab_files))
# data = pd.read_table(PAMlab_files[0], sep='\t',header=None,skiprows=1)
# data = data.ix[:,0:20]
# header = pd.read_table(PAMlab_files[0], sep='\t',header=None,nrows=1)
# header = header.ix[:,0:20]
# data.rename(columns=header)
