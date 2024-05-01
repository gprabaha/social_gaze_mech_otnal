#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:50:13 2024

@author: prabaha
"""

import os
import glob
import numpy as np
from scipy.io import loadmat
from math import degrees, atan2

import defaults

def get_root_data_dir(is_cluster):
    """
    Returns the root data directory based on whether it's running on a cluster or not.
    Parameters:
    - is_cluster (bool): Boolean flag indicating whether the program is running on a cluster.
    Returns:
    - root_data_dir (str): Root data directory path.
    """
    if is_cluster:
        root_data_dir = "/gpfs/milgram/project/chang/pg496/data_dir/otnal/"
    else:
        root_data_dir = "/Volumes/Stash/changlab/sorted_neural_data/social_gaze_otnal/AllFVProcessed/"
    return root_data_dir

def get_subfolders(root_dir):
    """
    Retrieves subfolders within a given directory.
    Parameters:
    - root_dir (str): Root directory path.
    Returns:
    - subfolders (list): List of subfolder paths.
    """
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    return subfolders

def extract_meta_info(session_paths):
    """
    Extracts meta-information from files in session paths.
    Parameters:
    - session_paths (list): List of paths to sessions.
    Returns:
    - meta_info_list (list): List of dictionaries containing meta-information for each session.
    """
    meta_info_list = []
    for session_path in session_paths:
        meta_info = {}
        file_list_info = glob.glob(f"{session_path}/*metaInfo.mat")
        file_list_runs = glob.glob(f"{session_path}/*runs.mat")
        if len(file_list_info) == 1:
            file_path_info = file_list_info[0]
            try:
                data_info = loadmat(file_path_info)
                info = data_info.get('info', None)
                # Selecting just the first run
                info = info[0][0]
                if info is not None:
                    monkey_1 = info['monkey_1'][0]
                    monkey_2 = info['monkey_2'][0]
                    OT_dose = float(info['OT_dose'][0])
                    NAL_dose = float(info['NAL_dose'][0])
                    meta_info.update({'monkey_1': monkey_1, 'monkey_2': monkey_2, 'OT_dose': OT_dose, 'NAL_dose': NAL_dose})
                else:
                    meta_info.update({'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None})
            except Exception as e:
                print(f"Error loading meta_info for folder: {session_path}: {e}")
                meta_info.update({'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None})
        else:
            print(f"Warning: No metaInfo found in folder: {session_path}.")
            meta_info.update({'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None})
        if len(file_list_runs) == 1:
            file_path_runs = file_list_runs[0]
            try:
                data_runs = loadmat(file_path_runs)
                runs = data_runs.get('runs', None)
                if runs is not None:
                    startS = [run['startS'][0][0] for run in runs[0]]
                    stopS = [run['stopS'][0][0] for run in runs[0]]
                    num_runs = len(startS)
                    meta_info.update({'startS': startS, 'stopS': stopS, 'num_runs': num_runs})
                else:
                    meta_info.update({'startS': None, 'stopS': None, 'num_runs': 0})
            except Exception as e:
                print(f"Error loading runs for folder: {session_path}: {e}")
                meta_info.update({'startS': None, 'stopS': None, 'num_runs': 0})
        else:
            print(f"Warning: No runs found in folder: {session_path}.")
            meta_info.update({'startS': None, 'stopS': None, 'num_runs': 0})
        meta_info_list.append(meta_info)
    return meta_info_list

def get_unique_doses(otnal_doses):
    """
    Finds unique rows and their indices in the given array.
    Parameters:
    - otnal_doses (ndarray): Input array.
    Returns:
    - unique_rows (ndarray): Unique rows in the input array.
    - indices_for_unique_rows (list): List of lists containing indices for each unique row.
    """
    unique_rows = np.unique(otnal_doses, axis=0)
    # Initialize an empty list to store indices for each unique row
    indices_for_unique_rows = []
    session_category = np.empty(otnal_doses.shape[0])
    session_category[:] = np.nan
    # Iterate over unique rows
    for i, row in enumerate(unique_rows):
        category = i
        # Find indices where each unique row occurs in the original array
        indices_for_row = np.where( (otnal_doses == row).all(axis=1))[0]
        session_category[indices_for_row] = category
        indices_for_unique_rows.append(indices_for_row.tolist())
    return unique_rows, indices_for_unique_rows, session_category

def px2deg(px, monitor_info=None):
    if monitor_info is None:
        monitor_info = defaults.fetch_monitor_info() # in defaults
    h = monitor_info['height']
    d = monitor_info['distance']
    r = monitor_info['vertical_resolution']
    deg_per_px = degrees(atan2(0.5 * h, d)) / (0.5 * r)
    deg = px * deg_per_px
    return deg

def create_timevec(n_samples, sampling_rate):
    return [i * sampling_rate for i in range(n_samples)]

def find_islands(binary_vec, min_samples=0):
    islands = []
    island_started = False
    island_start = 0
    for i, val in enumerate(binary_vec):
        if val == 1 and not island_started:
            island_started = True
            island_start = i
        elif val == 0 and island_started:
            island_started = False
            if i - island_start >= min_samples:
                islands.append([island_start, i - 1])
    # If the last island continues to the end of the array
    if island_started:
        if len(binary_vec) - island_start >= min_samples:
            islands.append([island_start, len(binary_vec) - 1])
    return np.array(islands)