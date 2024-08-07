#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:24:18 2024

@author: pg496
"""

import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import pickle

import util


def process_gaze_positions(dose_index_pairs, use_parallel, process_index, params):
    """
    Processes gaze positions either in parallel or serially.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - use_parallel (bool): Flag to determine if parallel processing should be used.
    - process_index (function): Function to process each index.
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    - params (dict): Updated dictionary of parameters.
    """
    if use_parallel:
        labelled_gaze_positions_m1 = process_gaze_positions_parallel(
            dose_index_pairs, process_index, params)
    else:
        labelled_gaze_positions_m1 = process_gaze_positions_serial(
            dose_index_pairs, process_index, params)
    return labelled_gaze_positions_m1, params


def process_gaze_positions_parallel(dose_index_pairs, process_index, params):
    """
    Processes gaze positions in parallel.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - process_index (function): Function to process each index.
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    - params (dict): Updated dictionary of parameters.
    """
    num_workers = min(multiprocessing.cpu_count(), len(dose_index_pairs))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_index, idx, params):
                   idx for _, idx in dose_index_pairs}
        results = []
        for future in tqdm(as_completed(futures),
                           desc="Processing gaze position for session",
                           unit="index", total=len(dose_index_pairs)):
            idx = futures[future]
            gaze_data = future.result()
            if gaze_data is not None:
                results.append((idx, gaze_data))
        results.sort(key=lambda x: x[0])
        labelled_gaze_positions_m1 = [gaze_data for _, gaze_data in results]
    return labelled_gaze_positions_m1, params


def process_gaze_positions_serial(dose_index_pairs, process_index, params):
    """
    Processes gaze positions serially.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - process_index (function): Function to process each index.
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    - params (dict): Updated dictionary of parameters.
    """
    labelled_gaze_positions_m1 = []
    for _, idx in tqdm(dose_index_pairs,
                       desc="Processing gaze position for session",
                       unit="index"):
        gaze_data = process_index(idx, params)
        if gaze_data is not None:
            labelled_gaze_positions_m1.append(gaze_data)
    return labelled_gaze_positions_m1, params



def save_labelled_gaze_positions(processed_data_dir, labelled_gaze_positions_m1, params):
    """
    Saves labelled gaze positions to a file.
    Parameters:
    - processed_data_dir (str): Directory to save processed data.
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    - params (dict): Dictionary of parameters.
    """
    flag_info = util.get_filename_flag_info(params)
    file_name = f'labelled_gaze_positions_m1{flag_info}.pkl'
    with open(os.path.join(processed_data_dir, file_name), 'wb') as f:
        pickle.dump(labelled_gaze_positions_m1, f)