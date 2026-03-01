from ecosound.soundscape.hmd import HMD
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues with Dask

if __name__ == '__main__':  # ← Add this line

    ## ####################################################################################################################

    deployment_dir = r'C:\Users\xavier.mouy\Documents\Projects\2025_Wellfleet\PBP_analysis_results\WellfleetHarbor_2023-07\NC'
    station_name = "Wellfleet"
    outdir = r'C:\Users\xavier.mouy\Documents\Projects\2025_Wellfleet\PBP_analysis_results\WellfleetHarbor_2023-07'
    nc_filename_pattern = r'_\d{8}\.nc$'
    start_date_string = '2023-07-19'
    end_date_string = '2023-09-27'
    integration_time = '1h'

    ## Get SPL values #####################################################################################################
    with HMD(n_workers=8, memory_per_worker='8GB', use_processes=True) as hmd_cluster:

        # load nc files
        hmd_cluster.load_nc_files(deployment_dir,recursive=True,time_range=(start_date_string, end_date_string),filename_pattern=nc_filename_pattern,prefilter_by_date=True, chunks={'time': 1440, 'frequency': -1})
        hmd_cluster.summary()

        # plot LTSA
        hmd_cluster.plot_ltsa(bin=integration_time,
                              #freq_range=(10,1000),
                              db_range=(32, 108),
                              #plot_date_range='fullyear',
                              scale='log',
                              #cmap='viridis',
                              statistic='median',
                              title=None,
                              figsize=(14, 6),
                              dpi=100,
                              save_path=os.path.join(outdir,f"LTSA_{station_name}_{start_date_string}_to_{end_date_string}.png"),
                              show=True,
                              return_data=False
                              )

        # plot PSD
        hmd_cluster.plot_psd(style='quantile',
                             #percentiles=[10, 50, 90],
                             freq_range=(100,10000),
                             db_range=(30,120),
                             scale='log',
                             cmap='binary',
                             legend_loc='outside right top',
                             linewidth=1.5,
                             alpha=1,
                             colors='Set2',
                             title=None,
                             figsize=(10, 6),
                             dpi=100,
                             save_path=os.path.join(outdir,f"PSD_{station_name}_{start_date_string}_to_{end_date_string}.png"),
                             show=False,
                             return_data=False,
                             )
