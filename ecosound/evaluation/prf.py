# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:48:18 2022

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.measurement import Measurement
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class PRF:
    def __init__(self, mode):
        pass

    @staticmethod
    def count(
        annot=None,
        detec=None,
        out_dir=None,
        target_class=None,
        files_to_use="both",  # 'detec', 'annot', 'both', list
        thresholds=np.arange(0, 1.05, 0.05),
        date_min=None,
        date_max=None,
        freq_ovp=True,
        dur_factor_max=None,
        dur_factor_min=None,
        ovlp_ratio_min=None,
        remove_duplicates=False,
        inherit_metadata=False,
        filter_deploymentID=False,
        do_plot=False,
        F_beta=1,
    ):

        # filter dates (if dataset partially annotated)
        if date_min:
            annot.filter("time_min_date >= '" + date_min + "'", inplace=True)
            detec.filter("time_min_date >= '" + date_min + "'", inplace=True)
        if date_max:
            annot.filter("time_max_date <= '" + date_max + "'", inplace=True)
            detec.filter("time_max_date <= '" + date_max + "'", inplace=True)

        # filter to the class of interest
        annot.filter('label_class == "' + target_class + '"', inplace=True)
        detec.filter('label_class == "' + target_class + '"', inplace=True)

        # Define list of files to use for the performance evaluation.
        if (
            files_to_use == "detec"
        ):  # case 1: all files with detections are used
            files_list = list(set(detec.data.audio_file_name))
        elif (
            files_to_use == "annot"
        ):  # case 2: all files with annotations are used
            files_list = list(set(annot.data.audio_file_name))
        elif (
            files_to_use == "both"
        ):  # case 3: all files with annotations or detections are used
            files_list1 = list(set(annot.data.audio_file_name))
            files_list2 = list(set(detec.data.audio_file_name))
            files_list = list(set(files_list1 + files_list2))

        elif (
            type(files_to_use) is list
        ):  # case 4: only files provided by the user are used
            files_list = files_to_use
        files_list.sort()

        # filter annotations with selected files to use
        annot.filter(
            "audio_file_name in @files_list",
            files_list=files_list,
            inplace=True,
        )

        # filter detections with selected files to use
        detec.filter(
            "audio_file_name in @files_list",
            files_list=files_list,
            inplace=True,
        )

        # loop through thresholds
        for th_idx, threshold in enumerate(thresholds):
            print("Threshold value: ", threshold)
            # filter detections for that threshold value
            detec_conf = detec.filter(
                "confidence >= " + str(threshold), inplace=False
            )
            # init
            FP = np.zeros(len(files_list))
            TP = np.zeros(len(files_list))
            FN = np.zeros(len(files_list))
            # loop through each file
            # for idx, file in enumerate(
            #     tqdm(files_list, desc="File", leave=True, miniters=1, colour="red")
            # ):
            # for idx, file in enumerate(files_list):
            for idx, file in enumerate(
                tqdm(
                    files_list,
                    desc="Progress",
                    leave=True,
                    miniters=1,
                    colour="green",
                )
            ):
                # print(idx)
                # filter to only keep data from this file
                annot_tmp = annot.filter("audio_file_name == '" + file + "'")
                detec_tmp = detec_conf.filter(
                    "audio_file_name=='" + file + "'", inplace=False
                )

                # count FP, TP, FN:
                if (
                    len(annot_tmp) == 0
                ):  # if no annotations -> FP = nb of detections
                    FP[idx] = len(detec_tmp)
                elif (
                    len(detec_tmp) == 0
                ):  # if no detections -> FN = nb of annotations
                    FN[idx] = len(annot_tmp)
                else:
                    ovlp = annot_tmp.filter_overlap_with(
                        detec_tmp,
                        freq_ovp=freq_ovp,
                        dur_factor_max=dur_factor_max,
                        dur_factor_min=dur_factor_min,
                        ovlp_ratio_min=ovlp_ratio_min,
                        remove_duplicates=remove_duplicates,
                        inherit_metadata=inherit_metadata,
                        filter_deploymentID=filter_deploymentID,
                        inplace=False,
                    )
                    FN[idx] = len(annot_tmp) - len(ovlp)
                    TP[idx] = len(ovlp)
                    FP[idx] = len(detec_tmp) - len(ovlp)

                # Sanity check
                if FP[idx] + TP[idx] != len(detec_tmp):
                    raise Exception(
                        "FP and TP don't add up to the total number of detections"
                    )
                elif TP[idx] + FN[idx] != len(annot_tmp):
                    raise Exception(
                        "FP and FN don't add up to the total number of annotations"
                    )

                # plot annot and detec boxes
                if do_plot:
                    PRF._plot_annot_boxes(
                        [annot_tmp, detec_tmp],
                        line_width=2,
                        colors=["blue", "red"],
                        labels=["Annotations", "Detections"],
                        title=file
                        + "\n TP:"
                        + str(TP[idx])
                        + " - FP: "
                        + str(FP[idx])
                        + " - FN: "
                        + str(FN[idx]),
                    )
            # create df for each file with threshold, TP, FP, FN
            tmp = pd.DataFrame(
                {
                    "file": files_list,
                    "threshold": threshold,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                }
            )
            if th_idx == 0:
                performance_per_file_count = tmp
            else:
                performance_per_file_count = pd.concat(
                    [performance_per_file_count, tmp], ignore_index=True
                )

            # Calculate P, R, and F for that confidence threshold
            perf_tmp = performance_per_file_count.query(
                "threshold ==" + str(threshold)
            )
            TP_th = perf_tmp["TP"].sum()
            FP_th = perf_tmp["FP"].sum()
            FN_th = perf_tmp["FN"].sum()
            R_th = TP_th / (TP_th + FN_th)
            if (TP_th + FP_th) == 0:
                P_th = 1
            else:
                P_th = TP_th / (TP_th + FP_th)
            F_th = (1 + F_beta**2) * ((P_th * R_th) / ((F_beta**2 * P_th) + R_th))
            tmp_PRF = pd.DataFrame(
                {
                    "threshold": [threshold],
                    "TP": [TP_th],
                    "FP": [FP_th],
                    "FN": [FN_th],
                    "R": [R_th],
                    "P": [P_th],
                    "F": [F_th],
                }
            )
            # Sanity check
            if FP_th + TP_th != len(detec_conf):
                raise Exception(
                    "FP and TP don't add up to the total number of detections"
                )
            elif TP_th + FN_th != len(annot):
                raise Exception(
                    "FP and FN don't add up to the total number of annotations"
                )

            if th_idx == 0:
                performance_PRF = tmp_PRF
            else:
                performance_PRF = pd.concat(
                    [performance_PRF, tmp_PRF], ignore_index=True
                )
        # plot PRF curves
        fig, ax = plt.subplots(1, 2)
        # ax[0].axis('equal')
        ax[0].plot(performance_PRF["P"], performance_PRF["R"], "k")
        ax[0].set_xlabel("Precision")
        ax[0].set_ylabel("Recall")
        ax[0].set(xlim=(0, 1), ylim=(0, 1))
        ax[0].grid()
        ax[0].set_aspect("equal", "box")
        fig.tight_layout()

        # ax[1].axis('equal')
        ax[1].plot(
            performance_PRF["threshold"],
            performance_PRF["P"],
            ":k",
            label="Precision",
        )
        ax[1].plot(
            performance_PRF["threshold"],
            performance_PRF["R"],
            "--k",
            label="Recall",
        )
        ax[1].plot(
            performance_PRF["threshold"],
            performance_PRF["F"],
            "k",
            label="$F_" + str(F_beta) + "$-score",
        )
        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("Score")
        ax[1].set(xlim=[0, 1], ylim=[0, 1])
        ax[1].grid()
        ax[1].legend()
        ax[1].set_aspect("equal", "box")
        fig.set_size_inches(8, 4)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Performance_graph.png"))

        # save CSV files
        performance_PRF.to_csv(
            os.path.join(out_dir, "Performance_full_dataset.csv"), index=False
        )
        performance_per_file_count.to_csv(
            os.path.join(out_dir, "Performance_per_file.csv"), index=False
        )

    @staticmethod
    def presence(
        annot=None,
        detec=None,
        out_dir=None,
        target_class=None,
        files_to_use="both",  # 'detec', 'annot', 'both', list, None
        thresholds=np.arange(0, 1.05, 0.05),
        date_min=None,
        date_max=None,
        F_beta=1,
        integration_time="1D",
        min_detec_nb=1,
    ):

        # # load ground truth data
        # annot = Annotation()
        # annot.from_netcdf(annot_file)

        # # annot = Annotation()
        # # annot.from_raven(
        # #     annot_file,
        # #     class_header="Call Type",
        # #     subclass_header="Overlap",
        # #     verbose=True,
        # # )

        # # load destections
        # detec = Annotation()
        # detec.from_sqlite(detec_file)

        # filter dates (if dataset partially annotated)
        if date_min:
            annot.filter("time_min_date >= '" + date_min + "'", inplace=True)
            detec.filter("time_min_date >= '" + date_min + "'", inplace=True)
        if date_max:
            annot.filter("time_max_date <= '" + date_max + "'", inplace=True)
            detec.filter("time_max_date <= '" + date_max + "'", inplace=True)

        # filter to the class of interest
        annot.filter('label_class == "' + target_class + '"', inplace=True)
        detec.filter('label_class == "' + target_class + '"', inplace=True)

        # Define list of files to use for the performance evaluation.
        if files_to_use:
            if (
                files_to_use == "detec"
            ):  # case 1: all files with detections are used
                files_list = list(set(detec.data.audio_file_name))
            elif (
                files_to_use == "annot"
            ):  # case 2: all files with annotations are used
                files_list = list(set(annot.data.audio_file_name))
            elif (
                files_to_use == "both"
            ):  # case 3: all files with annotations or detections are used
                files_list1 = list(set(annot.data.audio_file_name))
                files_list2 = list(set(detec.data.audio_file_name))
                files_list = list(set(files_list1 + files_list2))

            elif (
                type(files_to_use) is list
            ):  # case 4: only files provided by the user are used
                files_list = files_to_use
            files_list.sort()

            # filter annotations with selected files to use
            annot.filter(
                "audio_file_name in @files_list",
                files_list=files_list,
                inplace=True,
            )

            # filter detections with selected files to use
            detec.filter(
                "audio_file_name in @files_list",
                files_list=files_list,
                inplace=True,
            )

        # Create annootation aggregate
        aggr_min_date = min(
            min(annot.data["time_max_date"]), min(detec.data["time_max_date"])
        )
        aggr_max_date = max(
            max(annot.data["time_max_date"]), max(detec.data["time_max_date"])
        )
        annot_aggr = annot.calc_time_aggregate_1D(
            integration_time=integration_time,
            is_binary=False,
            start_date=aggr_min_date,
            end_date=aggr_max_date,
        )

        dates_list = annot_aggr.index

        # loop through thresholds
        # for th_idx, threshold in enumerate(
        #     tqdm(thresholds, desc="Progress", leave=True, miniters=1, colour="green")
        # ):
        for th_idx, threshold in enumerate(thresholds):
            print("Threshold value: ", threshold)
            # filter detections for that threshold value
            detec_conf = detec.filter(
                "confidence >= " + str(threshold), inplace=False
            )

            # Create detection aggregate
            detec_aggr = detec_conf.calc_time_aggregate_1D(
                integration_time=integration_time,
                is_binary=False,
                start_date=aggr_min_date,
                end_date=aggr_max_date,
            )

            # init
            FP = np.zeros(len(dates_list))
            TP = np.zeros(len(dates_list))
            FN = np.zeros(len(dates_list))
            # loop through each file
            # for idx, file in enumerate(
            #     tqdm(files_list, desc="File", leave=True, miniters=1, colour="red")
            # ):
            # for idx, file in enumerate(files_list):
            for idx, date in enumerate(
                tqdm(
                    dates_list,
                    desc="Progress",
                    leave=True,
                    miniters=1,
                    colour="green",
                )
            ):
                # filter to only keep data from this time frame
                try:
                    annot_tmp = annot_aggr.loc[date].value
                    detec_tmp = detec_aggr.loc[date].value
                except:
                    print("stop")

                # apply threshold on min number of detections
                if detec_tmp >= min_detec_nb:
                    detec_tmp = True
                else:
                    detec_tmp = False
                if annot_tmp >= 1:
                    annot_tmp = True
                else:
                    annot_tmp = False

                # count FP, TP, FN:
                if (annot_tmp is True) & (detec_tmp is True):
                    TP[idx] = 1
                elif (annot_tmp is True) & (detec_tmp is False):
                    FN[idx] = 1
                elif (annot_tmp is False) & (detec_tmp is True):
                    FP[idx] = 1
                elif (annot_tmp is False) & (detec_tmp is False):
                    pass

            # create df for each file with threshold, TP, FP, FN
            tmp = pd.DataFrame(
                {
                    "time": dates_list,
                    "threshold": threshold,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                }
            )
            if th_idx == 0:
                performance_per_interval_count = tmp
            else:
                performance_per_interval_count = pd.concat(
                    [performance_per_interval_count, tmp], ignore_index=True
                )

            # Calculate P, R, and F for that confidence threshold
            perf_tmp = performance_per_interval_count.query(
                "threshold ==" + str(threshold)
            )
            TP_th = perf_tmp["TP"].sum()
            FP_th = perf_tmp["FP"].sum()
            FN_th = perf_tmp["FN"].sum()
            R_th = TP_th / (TP_th + FN_th)
            if (TP_th + FP_th) == 0:
                P_th = 1
            else:
                P_th = TP_th / (TP_th + FP_th)
            F_th = (1 + F_beta**2) * (
                (P_th * R_th) / ((F_beta**2 * P_th) + R_th)
            )
            tmp_PRF = pd.DataFrame(
                {
                    "threshold": [threshold],
                    "TP": [TP_th],
                    "FP": [FP_th],
                    "FN": [FN_th],
                    "R": [R_th],
                    "P": [P_th],
                    "F": [F_th],
                }
            )
            # Sanity check
            # if FP_th + TP_th != len(detec_conf):
            #     raise Exception(
            #         "FP and TP don't add up to the total number of detections"
            #     )
            # elif TP_th + FN_th != len(annot):
            #     raise Exception(
            #         "FP and FN don't add up to the total number of annotations"
            #     )

            if th_idx == 0:
                performance_PRF = tmp_PRF
            else:
                performance_PRF = pd.concat(
                    [performance_PRF, tmp_PRF], ignore_index=True
                )

        # plot PRF curves
        fig, ax = plt.subplots(1, 2)
        # ax[0].axis('equal')
        ax[0].plot(performance_PRF["P"], performance_PRF["R"], "k")
        ax[0].set_xlabel("$Precision\ _{" + integration_time + "}$")
        ax[0].set_ylabel("$Recall\ _{" + integration_time + "}$")
        ax[0].set(xlim=(0, 1), ylim=(0, 1))
        ax[0].grid()
        ax[0].set_aspect("equal", "box")
        fig.tight_layout()

        # ax[1].axis('equal')
        ax[1].plot(
            performance_PRF["threshold"],
            performance_PRF["P"],
            ":k",
            label="$Precision\ _{" + integration_time + "}$",
        )
        ax[1].plot(
            performance_PRF["threshold"],
            performance_PRF["R"],
            "--k",
            label="$Recall\ _{" + integration_time + "}$",
        )
        ax[1].plot(
            performance_PRF["threshold"],
            performance_PRF["F"],
            "k",
            label="$F_" + str(F_beta) + " score\ _{" + integration_time + "}$",
        )
        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("Score")
        ax[1].set(xlim=[0, 1], ylim=[0, 1])
        ax[1].grid()
        ax[1].legend()
        ax[1].set_aspect("equal", "box")
        fig.set_size_inches(8, 4)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                out_dir, "Performance_" + integration_time + "_graph.png"
            )
        )

        # save CSV files
        performance_PRF.to_csv(
            os.path.join(
                out_dir,
                "Performance_" + integration_time + "_full_dataset.csv",
            ),
            index=False,
        )
        performance_per_interval_count.to_csv(
            os.path.join(
                out_dir, "Performance_per_" + integration_time + ".csv"
            ),
            index=False,
        )

    @staticmethod
    def _plot_annot_boxes(
        annot_list,
        line_width=2,
        colors=["blue"],
        labels=["Annotations"],
        title="",
    ):
        fig, ax = plt.subplots()
        facecolor = "none"
        alpha = 1
        for idx, annot in enumerate(annot_list):
            patches = []
            for index, row in annot.data.iterrows():
                # plot annotation boxes in Time-frquency
                x = row["time_min_offset"]
                y = row["frequency_min"]
                width = row["duration"]
                height = row["frequency_max"] - row["frequency_min"]
                rect = Rectangle(
                    (x, y),
                    width,
                    height,
                    linewidth=line_width,
                    edgecolor=colors[idx],
                    facecolor=facecolor,
                    alpha=alpha,
                    label=labels[idx],
                )
                patches.append(rect)
            p = PatchCollection(
                patches,
                edgecolor=colors[idx],
                # label=labels[idx],
                facecolor=facecolor,
            )
            ax.add_collection(p)
            # ax.add_patch(rect)
        ax.set_xlabel("Times (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        ax.grid()
        ax.plot([1], [1])
        plt.show()
        ax.legend()
