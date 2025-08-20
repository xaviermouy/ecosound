from ecosound.core.annotation import Annotation

nc_file = r'C:\Users\xavier.mouy\Documents\GitHub\Haddock-detector\data\nc_conversion_test\detections_merged.nc'
out_dir = r'C:\Users\xavier.mouy\Documents\GitHub\Haddock-detector\data\nc_conversion_test\Raven_tables'

# load detecions from NC file
annot = Annotation()
annot.from_netcdf(nc_file)
annot.summary()

# Convert to Raven tables (1 txt file per audio file)
annot.to_raven(outdir=out_dir, single_file=False)

