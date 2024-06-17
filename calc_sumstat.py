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
import yaml
import calendar

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

newf = pd.read_csv("indata/mf/Salmon.csv")

# Adjustements for river Ness, merging of receivers at the same location and one receiver West-bound (483479)
newf["receiver"] = newf["receiver"].replace([483495, 483466], 483488)
newf["receiver"] = newf["receiver"].replace([483463], 483460)
newf["receiver"] = newf["receiver"].replace([483457], 483492)
newf = newf[newf["receiver"] != 483479]


first_ping = newf.drop_duplicates(subset="tagid")
tag_info = pd.read_csv("indata/mf/moray_tag_info_corrected.csv")

# merge newf to tag_info with tagid and Tag_ID
tag_info_full = pd.merge(
    first_ping, tag_info, left_on="tagid", right_on="Tag_ID", how="right"
)

tag_info_full = tag_info_full[tag_info_full["Spp"] == "Salmon"]
###########################
# get release datetime
# Get rid of one idiotic format of date
tag_info_full["Date"] = tag_info_full["Date"].apply(
    lambda x: (
        x if x[-4:] == "2019" else x.split("/")[1] + "/" + x.split("/")[0] + "/2019"
    )
)
# Fix data where no-one entered release date
tag_info_full["date_string"] = tag_info_full["Release Date"].fillna(
    tag_info_full["Date"].apply(
        lambda x: "-".join([x.split("/")[0], calendar.month_abbr[int(x.split("/")[1])]])
    )
)
tag_info_full["date_string"] = tag_info_full["date_string"].apply(lambda x: "2019-" + x)
tag_info_full["time_string"] = tag_info_full["Release Time"].apply(lambda x: x + ":00")
tag_info_full["combined_datetime"] = (
    tag_info_full["date_string"] + " " + tag_info_full["time_string"]
)

datetime_format = "%Y-%d-%b %H:%M:%S"
tag_info_full["release_datetime"] = pd.to_datetime(
    tag_info_full["combined_datetime"], format=datetime_format, errors="coerce"
)

wtf = tag_info_full[tag_info_full["release_datetime"].isnull()]
if len(wtf):
    raise ValueError("Some release times are not in the correct format")

start_times = dict()
for array in newf["array"].unique():
    try:
        start_times[array] = tag_info_full[tag_info_full["array"] == array][
            "release_datetime"
        ].min()
        print(f"Starting time for array {array} is {start_times[array]}")
    except:
        continue

smolt_starts = dict()
for array in newf["array"].unique():
    try:
        this_array_list = []
        for smolt in tag_info_full["Tag_ID"][tag_info_full["array"] == array]:
            start_delay = (
                tag_info_full[tag_info_full["Tag_ID"] == smolt][
                    "release_datetime"
                ].values[0]
                - start_times[array]
            ).total_seconds() / 60
            this_array_list.append(int(start_delay))
        smolt_starts[array] = this_array_list
        # How many unique values we have?
        print(
            f"Array {array} has {len(set(this_array_list))} unique start times for {len(this_array_list)} smolts"
        )
    except:
        continue

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
    # distance in meters:
    rec_table["distance"] = rec_table["distance"] * 1000
    rec_table["distance_diff"] = rec_table["distance_diff"] * 1000
    pings_data0 = newf[newf["array"] == array]
    # merge on receiver
    pings_data = pd.merge(pings_data0, rec_table, on="receiver")

    # NOTE for some reason my efficency is often lower than that from the tables. It is strange since, without potential marine data, if anything mine should be higher.
    for rec in pings_data["receiver"].unique():
        # get number of unique smolts detected by this receiver
        tr_smolts = set(pings_data[pings_data["receiver"] == rec]["tagid"].unique())
        tr_dist = pings_data[pings_data["receiver"] == rec]["distance"].values[0]
        # get number of unique detections at receivers with distance greater than this receiver
        further_smolts = set(
            pings_data[pings_data["distance"] > tr_dist]["tagid"].unique()
        )
        tr_missed = further_smolts - tr_smolts
        # tr_unique = tr_smolts - further_smolts # if I include this in denominator it'll make it lower still
        if len(further_smolts) > 0:
            tr_efficency = 100 * (1 - (len(tr_missed) / len(further_smolts)))
        else:  # IF unknown from our computations use the receiver efficiency
            tr_efficency = pings_data[pings_data["receiver"] == rec][
                "efficiency"
            ].values[0]
        pings_data.loc[pings_data["receiver"] == rec, "mk_efficiency"] = tr_efficency

    # How many NaNs we have in distance records as a percentage of all?
    print(
        f'Array {array} has {pings_data["distance_diff"].isna().sum()/len(pings_data)*100:.2f}% of distance records missing'
    )
    # print pings_data efficency and mk_efficiency for all unique receivers
    print(pings_data.groupby("receiver")[["efficiency", "mk_efficiency"]].first())
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
    # get receiver 'distance' and 'efficency'
    rec_info = (
        the_array.groupby(["receiver"])
        .first()
        .reset_index()[["receiver", "distance", "efficiency"]]
    )
    detectors_dists = [int(z) for z in rec_info["distance"].values]
    detectors_probs = [float(z / 100) for z in rec_info["efficiency"].values]

    # We are redefining minutes_of_journey as the time since the first smolt in the run has been released
    # the_array["minutes_of_journey"] = the_array.groupby("tagid")["datetime"].transform(
    #     lambda x: (x - x.min()).dt.total_seconds() / 60
    # )
    the_array["minutes_of_journey"] = the_array.apply(
        lambda x: (x["datetime"] - start_times[array]).total_seconds() / 60, axis=1
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
    else:
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
        arrtimedian = np.nanmedian(mid_time, axis=0)

        meanspeed = np.nanmean(np.array(detlocs[1:]) / arrtimemean[1:])
        medianspeed = np.nanmean(np.array(detlocs[1:]) / arrtimedian[1:])

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
        # NOTE WITHOUT std deviation of arrival time!!!!
        sumstat = np.concatenate(
            [
                group_npin,
                group_nufi,
                np.array([meanspeed]),
                np.array([medianspeed]),
                np.array([co_occurrences_corrected]),
            ]
        )
        # save sumstat to csv in out folder
        np.savetxt(f"out/sumstat_{array.lower()}.csv", sumstat, delimiter=",")
        # save details about this river system for simulations
        # keep dtypes simple
        river_data = {
            "name": str(array.lower()),
            "no_smolts": int(no_smolts),
            "NO_RECEIVERS": int(NO_RECEIVERS),
            "simtime_max": int(the_array["minutes_of_journey"].max()),
            "smolt_starts": np.repeat(
                0, no_smolts
            ).tolist(),  # HACK we don't have release times yet?
            "detectors_dists": detectors_dists,
            "detectors_probs": detectors_probs,
            "DET_GROUPS": DET_GROUPS.tolist(),
            "smolt_starts": smolt_starts[array],
        }
        with open(f"out/river_data_{array.lower()}.yaml", "w") as file:
            yaml.dump(river_data, file)
