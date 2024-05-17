#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:36:36 2024

@author: prabaha
"""

import numpy as np
from tqdm import tqdm
import os
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import util
import load_data
import defaults
import fix

import pdb


### Function to extract meta-information from session paths
def extract_meta_info(session_paths, map_roi_coord_to_eyelink_space):
    """
    Extracts meta-information from files in session paths.
    Parameters:
    - session_paths (list): List of paths to sessions.
    Returns:
    - meta_info_list (list): List of dictionaries containing meta-information for each session.
    """
    meta_info_list = []
    for session_path in session_paths:
        dose_info = load_data.get_monkey_and_dose_data(session_path)
        if dose_info is not None:
            meta_info = {'session_name': os.path.basename(os.path.normpath(session_path))}
            meta_info.update(dose_info)
            runs_info = load_data.get_runs_data(session_path)
            meta_info.update(runs_info)
            meta_info['roi_bb_corners'] = load_data.get_m1_roi_bounding_boxes(session_path, map_roi_coord_to_eyelink_space)
            meta_info_list.append(meta_info)
    return meta_info_list


### Function to get unique doses
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
    indices_for_unique_rows = []
    session_category = np.empty(otnal_doses.shape[0])
    session_category[:] = np.nan
    for i, row in enumerate(unique_rows):
        category = i
        indices_for_row = np.where( (otnal_doses == row).all(axis=1))[0]
        session_category[indices_for_row] = category
        indices_for_unique_rows.append(indices_for_row.tolist())
    return unique_rows, indices_for_unique_rows, session_category


def extract_labelled_gaze_positions_m1(
        root_data_dir, unique_doses, dose_inds, meta_info_list,
        session_paths, session_categories,
        map_gaze_pos_coord_to_eyelink_space, use_parallel=True):
    """
    Extracts labelled gaze positions from files associated with unique doses.
    Parameters:
    - root_data_dir (str): Root directory for data storage.
    - unique_doses (ndarray): Unique dose combinations.
    - dose_inds (list): List of lists containing indices for each unique dose.
    - meta_info_list (list): List of dictionaries containing meta-information for each session.
    - session_paths (list): List of paths to sessions.
    - session_categories (ndarray): Session categories.
    - map_gaze_pos_coord_to_eyelink_space (bool): Flag to determine if coordinates should be remapped.
    - use_parallel (bool): Flag to determine if parallel processing should be used.
    Returns:
    - labelled_gaze_positions_m1 (list): List of tuples containing gaze positions and associated metadata.
    """

    def process_index(idx):
        folder_path = session_paths[idx]
        return load_data.get_labelled_gaze_positions_dict_m1(
            folder_path, meta_info_list, session_categories, idx,
            map_gaze_pos_coord_to_eyelink_space)
    
    labelled_gaze_positions_m1 = []
    dose_index_pairs = [(dose, idx) for dose, indices_list in zip(unique_doses, dose_inds) for idx in indices_list]
    if use_parallel:
        num_workers = min(multiprocessing.cpu_count(), len(dose_index_pairs))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_index, idx): idx for _, idx in dose_index_pairs}
            results = []
            for future in tqdm(as_completed(futures), desc="Processing gaze position for session", unit="index", total=len(dose_index_pairs)):
                idx = futures[future]
                gaze_data = future.result()
                if gaze_data is not None:
                    results.append((idx, gaze_data))
            # Sort results based on the original order of indices
            results.sort(key=lambda x: x[0])
            labelled_gaze_positions_m1 = [gaze_data for _, gaze_data in results]
    else:
        for _, idx in tqdm(dose_index_pairs, desc="Processing gaze position for session", unit="index"):
            gaze_data = process_index(idx)
            if gaze_data is not None:
                labelled_gaze_positions_m1.append(gaze_data)
    with open(os.path.join(root_data_dir, 'labelled_gaze_positions_m1.pkl'), 'wb') as f:
        pickle.dump(labelled_gaze_positions_m1, f) 
    return labelled_gaze_positions_m1



### Function to extract fixations with labels, possibly in parallel
def extract_fixations_with_labels_parallel(labelled_gaze_positions, root_data_dir, parallel=True):
    """
    Extracts fixations with labels, possibly in parallel.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    - parallel (bool): Whether to use parallel processing.
    Returns:
    - all_fixations (list): List of fixations.
    - all_fixation_labels (list): List of labels for fixations.
    """
    print("\nStarting to extract fixations:")
    session_identifiers = list(range(len(labelled_gaze_positions)))
    sessions = [(i, session[0], session[1]) for i, session in enumerate(labelled_gaze_positions)]
    if parallel:
        with Pool() as pool:
            results = list(tqdm(pool.imap(get_session_fixations, sessions),
                                total=len(sessions), desc="Extracting fixations in parallel", unit="session"))
    else:
        results = [get_session_fixations(session) for session in sessions]
    all_fixations = []
    all_fix_timepos = []
    all_fixation_labels = []
    for session_fixations, session_timepos_mat, session_labels in results:
        all_fixations.extend(session_fixations)
        all_fix_timepos.extend(session_timepos_mat)
        all_fixation_labels.extend(session_labels)
    col_names = ['category', 'session_id', 'session_name', 'run', 'block', 'fix_duration', 'mean_x_pos', 'mean_y_pos', 'fix_roi', 'agent']
    all_fixation_labels = pd.DataFrame(all_fixation_labels, columns=col_names)   
    np.save(os.path.join(root_data_dir, 'fixations_m1.npy'), all_fixations)
    np.save(os.path.join(root_data_dir, 'fixations_timepos_m1.npy'), all_fix_timepos)
    all_fixation_labels.to_csv(os.path.join(root_data_dir, 'fixation_labels_m1.csv'), index=False)
    return all_fixations, all_fix_timepos, all_fixation_labels


### Function to get fixations for a session
def get_session_fixations(session):
    """
    Extracts fixations for a session.
    Parameters:
    - session (tuple): Tuple containing session identifier, positions, and metadata.
    Returns:
    - fixations (list): List of fixations.
    - fixation_labels (list): List of labels for fixations.
    """
    session_identifier, positions, info = session
    session_name = info['session_name']
    sampling_rate = info['sampling_rate']
    n_samples = positions.shape[0]
    time_vec = util.create_timevec(n_samples, sampling_rate)
    category = info['category']
    n_runs = info['num_runs']
    n_intervals = n_runs - 1
    startS = info['startS']
    stopS = info['stopS']
    bbox_corners = info['roi_bb_corners']
    fix_timepos_mat, fix_vec_entire_session = fix.is_fixation(positions, time_vec, session_name, sampling_rate=sampling_rate)
    fixations = util.find_islands(fix_vec_entire_session)
    fixation_labels = []
    smallest_area_differences_for_roi_detection = []
    for start_stop in fixations:
        fix_duration = util.get_duration(start_stop)
        fix_positions = util.get_fix_positions(start_stop, positions)
        mean_fix_pos = np.nanmean(fix_positions, axis=0)
        run, block, fix_roi, smallest_diff = detect_run_block_and_roi(start_stop, startS, stopS, sampling_rate, mean_fix_pos, bbox_corners)
        smallest_area_differences_for_roi_detection.append(smallest_diff)
        agent = info['monkey_1']
        fixation_info = [category, session_identifier, session_name, run, block, fix_duration, mean_fix_pos[0], mean_fix_pos[1], fix_roi, agent]
        fixation_labels.append(fixation_info)
    assert fixations.shape[0] == len(fixation_labels)
    return fixations, fix_timepos_mat, fixation_labels


### Function to detect run, block, and ROI for a fixation
def detect_run_block_and_roi(start_stop, startS, stopS, sampling_rate, mean_fix_pos, bbox_corners):
    """
    Detects run, block, and ROI for a fixation.
    Parameters:
    - start_stop (tuple): Start and stop indices of fixation.
    - startS (list): List of start indices of runs.
    - stopS (list): List of stop indices of runs.
    - sampling_rate (float): Sampling rate.
    - mean_fix_pos (ndarray): Mean position of fixation.
    - bbox_corners (dict): Dictionary containing bounding boxes of ROIs.
    Returns:
    - run (int or None): Detected run number.
    - block (str): Detected block.
    - fix_roi (str): Detected ROI.
    """
    start, stop = start_stop*sampling_rate
    for i, (run_start, run_stop) in enumerate(zip(startS, stopS), start=1):
        if start >= run_start and stop <= run_stop:
            run = i
            block = 'mon_down'
            break
        elif i < len(startS) and stop <= startS[i]:
            run = None
            block = 'mon_up'
            break
    else:
        if start < startS[0] or stop > stopS[-1]:
            run = None
            block = 'discard'
        else:
            run = None
            block = 'discard'
    bounding_boxes = ['eye_bbox', 'face_bbox', 'left_obj_bbox', 'right_obj_bbox']
    smallest_diff = np.inf
    inside_all_roi = []
    for key in bounding_boxes:
        inside_roi, area_diff = util.is_inside_quadrilateral(mean_fix_pos, bbox_corners[key])
        inside_all_roi.append(inside_roi)
        smallest_diff = area_diff if area_diff < smallest_diff else smallest_diff
    if np.any(inside_all_roi):
        if inside_all_roi[0] and inside_all_roi[1]:
            fix_roi = bounding_boxes[0]
        else:
            fix_roi = bounding_boxes[bool(inside_all_roi)]
    else:
        fix_roi = 'out_of_roi' 
    return run, block, fix_roi, smallest_diff


def extract_spiketimes_for_all_sessions(root_data_dir, session_paths, is_parallel=True):
    spikeTs_s = []
    spikeTs_ms = []
    spikeTs_labels = []
    if is_parallel:
        # Process sessions in parallel with tqdm progress bar
        results = Parallel(n_jobs=-1)(delayed(
            load_data.get_spiketimes_and_labels_for_one_session)(session_path)
            for session_path in tqdm(session_paths, desc='Processing sessions'))
        # Iterate over results and concatenate
        for session_spikeTs_s, session_spikeTs_ms, session_spikeTs_labels \
            in tqdm(results, desc='Processing results'):
            spikeTs_s.extend(session_spikeTs_s)
            spikeTs_ms.extend(session_spikeTs_ms)
            spikeTs_labels.append(session_spikeTs_labels)
    else:
        # Process sessions sequentially
        for session_path in session_paths:
            session_spikeTs_s, session_spikeTs_ms, session_spikeTs_labels = load_data.get_spiketimes_and_labels_for_one_session(session_path)
            spikeTs_s.extend(session_spikeTs_s)
            spikeTs_ms.extend(session_spikeTs_ms)
            spikeTs_labels.append(session_spikeTs_labels)
    # Concatenate label dataframes
    if spikeTs_labels:
        all_labels = pd.concat(spikeTs_labels, ignore_index=True)
    else:
        all_labels = pd.DataFrame()
    # Check if spiketimes lists and labels have the same length
    if len(spikeTs_s) != len(all_labels):
        print("Warning: Length mismatch between spiketimes lists and labels.")
    # Save outputs to root_data_dir
    spiketimes_s_path = os.path.join(root_data_dir, 'spiketimes_s.pkl')
    spiketimes_ms_path = os.path.join(root_data_dir, 'spiketimes_ms.pkl')
    labels_path = os.path.join(root_data_dir, 'spike_labels.csv')
    # Save spikeTs_s as a pickle file
    with open(spiketimes_s_path, 'wb') as f:
        pickle.dump(spikeTs_s, f)
    # Save spikeTs_ms as a pickle file
    with open(spiketimes_ms_path, 'wb') as f:
        pickle.dump(spikeTs_ms, f)
    # Save arrays
    all_labels.to_csv(labels_path, index=False)
    return spikeTs_s, spikeTs_ms, all_labels













### Function to extract saccades with labels
def extract_saccades_with_labels(labelled_gaze_positions):
    """
    Extracts saccades with labels.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    Returns:
    - saccades (list): List of saccades.
    - saccade_labels (list): List of labels for saccades.
    """
    saccade_params = defaults.fetch_default_saccade_pars()
    vel_thresh = saccade_params['vel_thresh']
    min_samples = saccade_params['min_samples']
    smooth_func = saccade_params['smooth_func']
    saccades = []
    saccade_labels = []
    session_identifier = 0
    for session in tqdm(labelled_gaze_positions, desc="Extracting saccades for session", unit="session"):
        session_identifier += 1
        positions = session[0]
        info = session[1]
        sampling_rate = info['sampling_rate']
        n_samples = positions.shape[0]
        time_vec = util.create_timevec(n_samples, sampling_rate)
        category = info['category']
        n_runs = info['num_runs']
        for run in range(n_runs):
            run_start = info['startS'][run]
            run_stop = info['stopS'][run]
            run_time = (time_vec > run_start) & (time_vec <= run_stop)
            run_positions = positions[run_time,:]
            run_x = util.px2deg(run_positions[:,0].T)
            run_y = util.px2deg(run_positions[:,1].T)
            saccade_start_stops = find_saccades(run_x, run_y, sampling_rate, vel_thresh, min_samples, smooth_func)
            saccades_in_run = extract_saccade_positions(run_positions, saccade_start_stops)
            n_saccades = len(saccades_in_run)
            saccades.extend(saccades_in_run)
            saccade_labels.extend([[category, session_identifier, run]] * n_saccades)
    assert len(saccades) == len(saccade_labels)
    return saccades, saccade_labels


### Function to find saccades
def find_saccades(x, y, sr, vel_thresh, min_samples, smooth_func):
    """
    Finds saccades.
    Parameters:
    - x (array-like): x-coordinates of eye movements.
    - y (array-like): y-coordinates of eye movements.
    - sr (float): Sampling rate.
    - vel_thresh (float): Minimum velocity threshold for saccade onset.
    - min_samples (int): Minimum duration of a saccade in samples.
    - smooth_func (function): Function for smoothing input data.
    Returns:
    - start_stops (list): List of start and stop indices of saccades for each trial.
    """
    assert x.shape == y.shape
    start_stops = []
    x0 = smooth_func(x)
    y0 = smooth_func(y)
    vx = np.gradient(x0) / sr
    vy = np.gradient(y0) / sr
    vel_norm = np.sqrt(vx**2 + vy**2)  # Norm of velocity vector
    above_thresh = (vel_norm >= vel_thresh[0]) & (vel_norm <= vel_thresh[1])
    start_stops = util.find_islands(above_thresh, min_samples)
    return start_stops


### Function to extract saccade positions
def extract_saccade_positions(run_positions, saccade_start_stops):
    """
    Extracts saccade positions.
    Parameters:
    - run_positions (ndarray): Array containing gaze positions for a run.
    - saccade_start_stops (list): List of start and stop indices of saccades.
    Returns:
    - saccades (list): List of saccades.
    """
    saccades = []
    for start, stop in saccade_start_stops:
        saccade = run_positions[start:stop+1, :]
        saccades.append(saccade)
    return saccades
