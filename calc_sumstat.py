import csv
import glob
import pickle
from itertools import cycle

import dateutil
import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
from matplotlib import cm
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

newf = pd.read_csv("indata/mf/Salmon.csv")

rivsys = []
arrays = dict()
for array in newf["array"].unique():
    # get receiver locations
    try:
        rec_table = pd.read_csv(f"rec_data/{array.lower()}.csv")

    except:
        print(f"Array detail {array} not found")
        continue
    """
    ['Receiver',
    'Beat/Location',
    'Distance (km)',
    'Distance diff. (km)',
    'Receiver efficiency (%)',
    'Confirmed Survival (%)',
    '% losses per km',
    'Median residences (mins)',
    'Median ROM (m/s)',
    'Median travel duration (days)']
    """
    rec_table.columns = [
        "receiver",
        "location_name",
        "distance",
        "distance_diff",
        "efficiency",
        "survival",
        "losses",
        "residence",
        "rom",
        "duration",
    ]
    # remove `*` character from receiver names
    rec_table["receiver"] = rec_table["receiver"].str.replace("*", "")
    # remove entries where 'receiver' is not a number
    rec_table = rec_table[rec_table["receiver"].str.isnumeric()]
    # convert 'receiver' to integer
    rec_table["receiver"] = rec_table["receiver"].astype(int)
    pings_data0 = newf[newf["array"] == array]
    # merge on receiver
    pings_data = pd.merge(pings_data0, rec_table, on="receiver")
    # How many NaNs we have in distance records as a percentage of all?
    print(
        f'Array {array} has {pings_data["distance_diff"].isna().sum()/len(pings_data)*100:.2f}% of distance records missing'
    )
    arrays[array] = pings_data
    rivsys.append(array)


for array in rivsys:
    print(f"Processing array {array}")
    the_array = arrays[array]
    recids = the_array["receiver"].unique()
    smoltids = the_array["tagid"].unique()
    dists = the_array["distance"].unique()

    # For each 'tagid' calculate time as a delta in minutes from the very first detection
    the_array["datetime"] = pd.to_datetime(the_array["datetime"])
    # order the array by distance - so that receiver data is in order
    the_array = the_array.sort_values(by="distance")
    # Determine the unique order of 'receiver' as they appear
    unique_receivers = the_array["receiver"].unique()
    the_array["receiver"] = pd.Categorical(
        the_array["receiver"], categories=unique_receivers, ordered=True
    )

    the_array["minutes_of_journey"] = the_array.groupby("tagid")["datetime"].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60
    )
    detlocs = the_array["distance"].unique()

    # For each 'tagid' get, time of first, last and number of detections at each receiver
    first_det = np.array(
        the_array.groupby(["tagid", "receiver"])["minutes_of_journey"].min().unstack()
    )
    last_det = np.array(
        the_array.groupby(["tagid", "receiver"])["minutes_of_journey"].max().unstack()
    )
    cumu_det = np.array(
        the_array.groupby(["tagid", "receiver"])["minutes_of_journey"]
        .count()
        .unstack()
        .fillna(0)
    )

    NO_RECEIVERS = the_array["receiver"].nunique()
    if NO_RECEIVERS < 3:
        print(f"Skipping array {array} as it has less than 3 receivers")
        continue
    no_smolts = the_array["tagid"].nunique()

    # divide receivers into three groups, writing the group number into the array
    DET_GROUPS = np.concatenate(
        [
            np.repeat(0, NO_RECEIVERS // 3),
            np.repeat(1, NO_RECEIVERS // 3 + NO_RECEIVERS % 3),
            np.repeat(2, NO_RECEIVERS // 3),
        ]
    )

    # Assuming first_det and last_det are now 2D: [no_smolts, NO_RECEIVERS]
    first_det_expanded = first_det[
        :, np.newaxis, :
    ]  # Shape: [no_smolts, 1, NO_RECEIVERS]
    last_det_expanded = last_det[
        :, np.newaxis, :
    ]  # Shape: [no_smolts, 1, NO_RECEIVERS]

    # Broadcasting to compare all pairs
    overlaps = np.logical_and(
        first_det_expanded
        <= last_det[np.newaxis, :, :],  # First of one before last of other
        last_det_expanded
        >= first_det[np.newaxis, :, :],  # Last of one after first of other
    )

    for receiver_index in range(NO_RECEIVERS):
        np.fill_diagonal(overlaps[:, :, receiver_index], False)

    # Summing the boolean overlaps will give us the count of co-occurrences for each pair
    co_occurrences = np.sum(overlaps)  # Sum over smolts and receivers

    # We only want to count each pair once, so we subtract the count of smolts themselves and divide by 2
    co_occurrences_corrected = co_occurrences / (2 * no_smolts * NO_RECEIVERS)

    # Additional calculations are modified to remove batch processing
    npin = np.ma.masked_equal(cumu_det, 0).mean(axis=0).data
    mid_time = ((last_det - first_det) / 2) + first_det

    nufi = np.count_nonzero(cumu_det, axis=0) / no_smolts
    arrtimestd = np.nanstd(mid_time, axis=0) / detlocs
    arrtimemean = np.nanmean(mid_time, axis=0)

    meanspeed = 1 / (30 * np.nanmean(arrtimemean[1:] * (1 / np.array(detlocs[1:]))))

    group_npin = np.zeros(3)
    group_nufi = np.zeros(3)
    group_arrtimestd = np.zeros(3)

    for zzz in range(3):
        indices = np.where(np.array(DET_GROUPS) == zzz)
        group_start = indices[0][0]
        group_end = indices[0][-1]
        this_group_npin = npin[group_start : group_end + 1]
        this_group_nufi = nufi[group_start : group_end + 1]
        this_group_arrtimestd = arrtimestd[group_start : group_end + 1]
        group_npin[zzz] = np.mean(this_group_npin)
        group_nufi[zzz] = np.mean(this_group_nufi)
        group_arrtimestd[zzz] = np.mean(this_group_arrtimestd)

    # Prepare the summary statistics
    sumstat = np.concatenate(
        [
            group_npin,
            group_nufi,
            group_arrtimestd,
            np.array([meanspeed]),
            np.array([co_occurrences_corrected]),
        ]
    )
    # save sumstat to csv in out folder
    np.savetxt(f"out/sumstat_{array.lower()}.csv", sumstat, delimiter=",")
