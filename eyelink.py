#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:24:18 2024

@author: pg496
"""

import multiprocessing
from tqdm import tqdm
import os
import pickle

import util
import load_data

import pdb


def process_gaze_positions(dose_index_pairs, use_parallel, params):
    """
    Processes gaze positions either in parallel or serially.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - use_parallel (bool): Flag to determine if parallel processing should be used.
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    - params (dict): Updated dictionary of parameters.
    """
    if use_parallel:
        labelled_gaze_positions_m1 = process_gaze_positions_parallel(
            dose_index_pairs, params)
    else:
        labelled_gaze_positions_m1 = process_gaze_positions_serial(
            dose_index_pairs, params)
    return labelled_gaze_positions_m1, params


def process_gaze_positions_parallel(dose_index_pairs, params):
    """
    Processes gaze positions in parallel.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    - params (dict): Updated dictionary of parameters.
    """
    num_processes = min(multiprocessing.cpu_count(), len(dose_index_pairs))
    print(f'Gaze sig num processes: {num_processes}')
    with multiprocessing.Pool(processes=num_processes) as pool:
        labelled_gaze_positions_m1 = [
            result for result in tqdm(
                pool.imap_unordered(
                    process_index_func_wrapper, 
                    [(idx, params) for _, idx in dose_index_pairs]
                ),
                desc="Processing gaze position in parallel", 
                unit="index", 
                total=len(dose_index_pairs)
            ) if result is not None
        ]
    return labelled_gaze_positions_m1, params


def process_gaze_positions_serial(dose_index_pairs, params):
    """
    Processes gaze positions serially.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    - params (dict): Updated dictionary of parameters.
    """
    labelled_gaze_positions_m1 = []
    for _, idx in tqdm(dose_index_pairs,
                       desc="Processing gaze position in serial",
                       unit="index"):
        gaze_data = process_index_func_wrapper((idx, params))
        pdb.set_trace()
        if gaze_data is not None:
            labelled_gaze_positions_m1.append(gaze_data)
    return labelled_gaze_positions_m1, params


def process_index_func_wrapper(args):
    idx, params = args
    return load_data.get_labelled_gaze_positions_dict_m1(idx, params)



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