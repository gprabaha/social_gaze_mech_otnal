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
def extract_meta_info(params):
    """
    Extracts meta-information from files in session paths.
    Parameters:
    - session_paths (list): List of paths to sessions.
    Returns:
    - meta_info_list (list): List of dictionaries containing meta-information for each session.
    """
    meta_info_list = []
    for session_path in params['session_paths']:
        dose_info = load_data.get_monkey_and_dose_data(session_path)
        if dose_info is not None:
            meta_info = {'session_name':
                         os.path.basename(os.path.normpath(session_path))}
            meta_info.update(dose_info)
            runs_info = load_data.get_runs_data(session_path)
            meta_info.update(runs_info)
            meta_info['roi_bb_corners'] = \
                load_data.load_farplane_cal_and_get_bl_and_tr_roi_coords_m1(
                    session_path, params)
            meta_info_list.append(meta_info)
    return meta_info_list


### Function to get unique doses
def get_unique_doses(params):
    """
    Finds unique rows and their indices in the given array.
    Parameters:
    - otnal_doses (ndarray): Input array.
    Returns:
    - unique_rows (ndarray): Unique rows in the input array.
    - indices_for_unique_rows (list): List of lists containing indices for each unique row.
    """
    otnal_doses = params['otnal_doses']
    unique_rows = np.unique(otnal_doses, axis=0)
    indices_for_unique_rows = []
    session_category = np.empty(otnal_doses.shape[0])
    session_category[:] = np.nan
    for i, row in enumerate(unique_rows):
        category = i
        indices_for_row = np.where( (otnal_doses == row).all(axis=1))[0]
        session_category[indices_for_row] = category
        indices_for_unique_rows.append(indices_for_row.tolist())
    params.update({'unique_doses': unique_rows,
                   'dose_inds': indices_for_unique_rows,
                   'session_categories': session_category})
    return params

###
def extract_labelled_gaze_positions_m1(params):
    """
    Extracts labelled gaze positions from files associated with unique doses.
    Parameters:
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of tuples containing gaze positions and associated metadata.
    """
    processed_data_dir = params['processed_data_dir']
    unique_doses = params.get('unique_doses')
    dose_inds = params.get('dose_inds')
    use_parallel = params.get('use_parallel', True)
    
    def process_index(idx):
        return load_data.get_labelled_gaze_positions_dict_m1(idx, params)
    
    dose_index_pairs = [(dose, idx) for dose, indices_list
                        in zip(unique_doses, dose_inds)
                        for idx in indices_list]
    labelled_gaze_positions_m1 = process_gaze_positions(
        dose_index_pairs, use_parallel, process_index)
    save_labelled_gaze_positions(
        processed_data_dir, labelled_gaze_positions_m1, params)
    return labelled_gaze_positions_m1


def process_gaze_positions(dose_index_pairs, use_parallel, process_index):
    """
    Processes gaze positions either in parallel or serially.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - use_parallel (bool): Flag to determine if parallel processing should be used.
    - process_index (function): Function to process each index.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    """
    if use_parallel:
        return process_gaze_positions_parallel(dose_index_pairs, process_index)
    else:
        return process_gaze_positions_serial(dose_index_pairs, process_index)


def process_gaze_positions_parallel(dose_index_pairs, process_index):
    """
    Processes gaze positions in parallel.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - process_index (function): Function to process each index.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    """
    num_workers = min(multiprocessing.cpu_count(), len(dose_index_pairs))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_index, idx):
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
        return [gaze_data for _, gaze_data in results]


def process_gaze_positions_serial(dose_index_pairs, process_index):
    """
    Processes gaze positions serially.
    Parameters:
    - dose_index_pairs (list): List of dose and index pairs.
    - process_index (function): Function to process each index.
    Returns:
    - labelled_gaze_positions_m1 (list): List of processed gaze positions.
    """
    labelled_gaze_positions_m1 = []
    for _, idx in tqdm(dose_index_pairs,
                       desc="Processing gaze position for session",
                       unit="index"):
        gaze_data = process_index(idx)
        if gaze_data is not None:
            labelled_gaze_positions_m1.append(gaze_data)
    return labelled_gaze_positions_m1


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


### Function to extract fixations with labels, possibly in parallel
def extract_fixations_with_labels_parallel(labelled_gaze_positions, params):
    """
    Extracts fixations with labels, possibly in parallel.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fixations (list): List of fixations.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - all_fixation_labels (pd.DataFrame): DataFrame of labels for fixations.
    """
    print("\nStarting to extract fixations:")
    use_parallel = params.get('use_parallel', True)
    all_fixations, all_fix_timepos, fix_detection_results = \
        extract_or_load_fixations(labelled_gaze_positions, params)
    all_fixation_labels = generate_fixation_labels(
        fix_detection_results, params, use_parallel)
    return all_fixations, all_fix_timepos, all_fixation_labels


def extract_or_load_fixations(labelled_gaze_positions, params):
    """
    Extract or load fixations based on parameters.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fixations (list): List of fixations.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    if params.get('remake_fixations', False):
        return extract_all_fixations_from_labelled_gaze_positions(
            labelled_gaze_positions, params)
    else:
        results_file_name = f'fixation_session_results_m1{flag_info}.npz'
        if os.path.exists(os.path.join(processed_data_dir, results_file_name)):
            return load_existing_fixations(params)
        else:
            return extract_all_fixations_from_labelled_gaze_positions(
                labelled_gaze_positions, params)


def load_existing_fixations(params):
    """
    Load existing fixations from files.
    Parameters:
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fixations (list): List of fixations.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    """
    all_fixations, all_fix_timepos = load_data.load_m1_fixations(params)
    fix_detection_results = load_data.load_fix_detection_results(params)
    return all_fixations, all_fix_timepos, fix_detection_results


def generate_fixation_labels(fix_detection_results, params, use_parallel):
    """
    Generate fixation labels based on detection results.
    Parameters:
    - fix_detection_results (list): List of fixation detection results.
    - params (dict): Dictionary of parameters.
    - use_parallel (bool): Whether to use parallel processing.
    Returns:
    - all_fixation_labels (pd.DataFrame): DataFrame of labels for fixations.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    if params.get('remake_fixation_labels', False):
        fixation_labels = parallel_generate_labels(fix_detection_results) \
            if use_parallel else serial_generate_labels(fix_detection_results)
        all_fixation_labels = []
        for session_labels in fixation_labels:
            all_fixation_labels.extend(session_labels)
        col_names = ['start_time', 'end_time', 'category', 'session_name',
                     'run', 'block', 'fix_duration', 'mean_x_pos', 'mean_y_pos',
                     'fix_roi', 'agent']
        all_fixation_labels = pd.DataFrame(
            all_fixation_labels, columns=col_names)
        fixation_labels_file_name = f'fixation_labels_m1{flag_info}.csv'
        all_fixation_labels.to_csv(os.path.join(
            processed_data_dir, fixation_labels_file_name), index=False)
    else:
        all_fixation_labels = load_data.load_m1_fixation_labels(params)
    return all_fixation_labels


def parallel_generate_labels(fix_detection_results):
    """
    Generate fixation labels in parallel.
    Parameters:
    - fix_detection_results (list): List of fixation detection results.
    Returns:
    - fixation_labels (list): List of generated labels.
    """
    print("\nGenerating fixation labels in parallel")
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, len(fix_detection_results))
    with Pool(num_processes) as pool:
        fixation_labels = pool.map(
            generate_session_fixation_labels, fix_detection_results)
    return fixation_labels


def serial_generate_labels(fix_detection_results):
    """
    Generate fixation labels serially.
    Parameters:
    - fix_detection_results (list): List of fixation detection results.
    Returns:
    - fixation_labels (list): List of generated labels.
    """
    print("\nGenerating fixation labels serially")
    return [generate_session_fixation_labels(result)
            for result in fix_detection_results]


def extract_all_fixations_from_labelled_gaze_positions(
        labelled_gaze_positions, params):
    """
    Extracts fixations from labelled gaze positions.
    Parameters:
    - labelled_gaze_positions (list): List of labelled gaze positions.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fixations (list): List of all fixations.
    - all_fix_timepos (pd.DataFrame): List of fixation time positions.
    """
    processed_data_dir = params['processed_data_dir']
    use_parallel = params.get('use_parallel', True)
    sessions_data = [(session_data[0], session_data[1], params)
                     for session_data in labelled_gaze_positions]
    fix_detection_results = extract_fixations(sessions_data, use_parallel)
    all_fixations, all_fix_timepos, fix_timepos_list, info_list = \
        process_fixation_results(fix_detection_results)
    save_fixation_results(
        processed_data_dir, fix_timepos_list, all_fixations,
        all_fix_timepos, info_list, params)
    return all_fixations, all_fix_timepos, fix_detection_results


def extract_fixations(sessions_data, use_parallel):
    """
    Extracts fixations from session data.
    Parameters:
    - sessions_data (list): List of session data tuples.
    - use_parallel (bool): Flag to determine if parallel processing should be used.
    Returns:
    - fix_detection_results (list): List of fixation detection results.
    """
    if use_parallel:
        print("\nExtracting fixations in parallel")
        num_cores = multiprocessing.cpu_count()
        num_processes = min(num_cores, len(sessions_data))
        with Pool(num_processes) as pool:
            fix_detection_results = pool.map(
                get_session_fixations, sessions_data)
    else:
        print("\nExtracting fixations serially")
        fix_detection_results = [get_session_fixations(session_data)
                                 for session_data in sessions_data]
    return fix_detection_results


def process_fixation_results(fix_detection_results):
    """
    Processes the results from fixation detection.
    Parameters:
    - fix_detection_results (list): List of fixation detection results.
    Returns:
    - all_fixations (list): List of all fixations.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_timepos_list (list): List of fixation time positions for each session.
    - info_list (list): List of metadata information for each session.
    """
    all_fixations = []
    all_fix_timepos = pd.DataFrame()
    fix_timepos_list = []
    info_list = []
    for session_fixations, session_timepos_df, info in fix_detection_results:
        all_fixations.extend(session_fixations)
        all_fix_timepos = pd.concat([all_fix_timepos,
                                     session_timepos_df], ignore_index=True)
        fix_timepos_list.append(session_timepos_df)
        info_list.append(info)
    return all_fixations, all_fix_timepos, fix_timepos_list, info_list


def save_fixation_results(processed_data_dir, fix_timepos_list,
                          all_fixations, all_fix_timepos, info_list, params):
    """
    Saves fixation results to files.
    Parameters:
    - processed_data_dir (str): Directory to save processed data.
    - fix_timepos_list (list): List of fixation time positions for each session.
    - all_fixations (list): List of all fixations.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - info_list (list): List of metadata information for each session.
    - params (dict): Dictionary of parameters.
    """
    flag_info = util.get_filename_flag_info(params)
    fix_timepos_file_name = f'fix_session_timepos_list_m1{flag_info}.pkl'
    with open(os.path.join(processed_data_dir, fix_timepos_file_name), 'wb') as f:
        pickle.dump(fix_timepos_list, f)
    fixations_list = np.array(all_fixations, dtype=object)
    info_list = np.array(info_list, dtype=object)
    results_file_name = f'fixation_session_results_m1{flag_info}.npz'
    np.savez(os.path.join(processed_data_dir, results_file_name),
             fixations=fixations_list, info=info_list)
    fixations_file_name = f'fixations_m1{flag_info}.npy'
    np.save(os.path.join(processed_data_dir, fixations_file_name),
            fixations_list)
    timepos_file_name = f'fix_timepos_m1{flag_info}.csv'
    all_fix_timepos.to_csv(os.path.join(processed_data_dir,
                                        timepos_file_name), index=False)


def get_session_fixations(session_data):
    """
    Extracts fixations for a session.
    Parameters:
    - session_data (tuple): Tuple containing session identifier, positions, and metadata.
    Returns:
    - fixations (list): List of fixations.
    - fixation_timepos_mat (pd.DataFrame): DataFrame of fixation time positions.
    - info (dict): Metadata information for the session.
    """
    positions, info, params = session_data
    session_name = info['session_name']
    sampling_rate = info['sampling_rate']
    n_samples = positions.shape[0]
    time_vec = util.create_timevec(n_samples, sampling_rate)
    fix_timepos_df, fix_vec_entire_session = fix.is_fixation(
        positions, time_vec, session_name, sampling_rate=sampling_rate)
    fixations = util.find_islands(fix_vec_entire_session)
    return fixations, fix_timepos_df, info


### Function to generate fixation labels for each session
def generate_session_fixation_labels(fix_detection_result):
    """
    Generates fixation labels for each session.
    Parameters:
    - fix_detection_result (tuple): Tuple containing fixation detection results and session information.
    Returns:
    - fixation_labels (list): List of fixation labels.
    """
    session_fixations, session_timepos_mat, info = fix_detection_result
    fixation_labels = []
    for row in tqdm(session_timepos_mat.itertuples(index=False),
                    desc=f"{info['session_name']}: n fixations labelled"):
        fix_x = row.fix_x
        fix_y = row.fix_y
        start_time = row.start_time
        end_time = row.end_time
        fix_duration = row.duration
        mean_fix_pos = [fix_x, fix_y]
        run, block, fix_roi = detect_run_block_and_roi(
            [start_time, end_time], info['startS'], info['stopS'],
            info['sampling_rate'], mean_fix_pos, info['roi_bb_corners'])
        fixation_info = [start_time, end_time, info['category'],
                         info['session_name'], run, block, fix_duration,
                         mean_fix_pos[0], mean_fix_pos[1], fix_roi, info['monkey_1']]
        fixation_labels.append(fixation_info)
    return fixation_labels


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
    start, stop = start_stop
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
    fix_roi = determine_fix_roi(mean_fix_pos, bbox_corners)
    return run, block, fix_roi


def determine_fix_roi(mean_fix_pos, bbox_corners):
    """
    Determines the fixation ROI based on mean position and bounding box corners.
    Parameters:
    - mean_fix_pos (ndarray): Mean position of fixation.
    - bbox_corners (dict): Dictionary containing bounding boxes of ROIs.
    Returns:
    - fix_roi (str): Detected ROI.
    """
    bounding_boxes = ['eye_bbox', 'face_bbox', 'left_obj_bbox', 'right_obj_bbox']
    inside_roi = [util.is_inside_roi(mean_fix_pos, bbox_corners[key])
                  for key in bounding_boxes]
    if any(inside_roi):
        if inside_roi[0] and inside_roi[1]:
            return bounding_boxes[0]
        return bounding_boxes[inside_roi.index(True)]
    return 'out_of_roi'


def extract_spiketimes_for_all_sessions(params):
    root_data_dir = params.get('root_data_dir')
    session_paths = params.get('session_paths')
    is_parallel = params.get('use_parallel', True)
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
            session_spikeTs_s, session_spikeTs_ms, session_spikeTs_labels = \
                load_data.get_spiketimes_and_labels_for_one_session(session_path)
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
    # Construct flag_info based on params
    flag_info = util.get_filename_flag_info(params)
    # Save outputs to root_data_dir with flag_info
    spiketimes_s_path = os.path.join(root_data_dir, f'spiketimes_s{flag_info}.pkl')
    spiketimes_ms_path = os.path.join(root_data_dir, f'spiketimes_ms{flag_info}.pkl')
    labels_path = os.path.join(root_data_dir, f'spike_labels{flag_info}.csv')
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
    for session in tqdm(labelled_gaze_positions,
                        desc="Extracting saccades for session", unit="session"):
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
            saccade_start_stops = find_saccades(
                run_x, run_y, sampling_rate,
                vel_thresh, min_samples, smooth_func)
            saccades_in_run = extract_saccade_positions(
                run_positions, saccade_start_stops)
            n_saccades = len(saccades_in_run)
            saccades.extend(saccades_in_run)
            saccade_labels.extend(
                [[category, session_identifier, run]] * n_saccades)
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
