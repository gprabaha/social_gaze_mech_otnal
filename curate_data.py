#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:36:36 2024

@author: prabaha
"""

import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import os
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import ast

import util
import load_data
import defaults
import fix

import pdb


### Function to extract meta-information and update params
def extract_and_update_meta_info(params):
    """
    Extracts meta-information from files in session paths and updates the params dictionary.
    Parameters:
    - params (dict): Dictionary containing session paths and other parameters.
    Returns:
    - params (dict): Updated dictionary with meta-information and dose arrays.
    """
    meta_info_list = []
    for session_path in params['session_paths']:
        dose_info = load_data.get_monkey_and_dose_data(session_path)
        if dose_info is not None:
            meta_info = {
                'session_name': os.path.basename(
                    os.path.normpath(session_path))}
            meta_info.update(dose_info)
            runs_info = load_data.get_runs_data(session_path)
            meta_info.update(runs_info)
            meta_info['roi_bb_corners'] = \
                load_data.load_farplane_cal_and_get_bl_and_tr_roi_coords_m1(
                    session_path, params)
            meta_info_list.append(meta_info)
    params['meta_info_list'] = meta_info_list
    otnal_doses = np.array(
        [[meta_info['OT_dose'], meta_info['NAL_dose']]
         for meta_info in meta_info_list], dtype=np.float64)
    params['otnal_doses'] = otnal_doses
    return params



### Function to get unique doses
def get_unique_doses(params):
    """
    Finds unique rows and their indices in the given array.
    Parameters:
    - otnal_doses (ndarray): Input array.
    Returns:
    - unique_rows (ndarray): Unique rows in the input array.
    - indices_for_unique_rows (list): List of lists containing indices for
    each unique row.
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
    - labelled_gaze_positions_m1 (list): List of tuples containing gaze
    positions and associated metadata.
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
        return process_gaze_positions_parallel(
            dose_index_pairs, process_index)
    else:
        return process_gaze_positions_serial(
            dose_index_pairs, process_index)


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


def extract_fixations_with_labels_parallel(labelled_gaze_positions, params):
    """
    Extracts fixations with labels, possibly in parallel.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions
    and associated metadata.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fixation_labels (pd.DataFrame): DataFrame of labels for fixations.
    """
    print("\nStarting to extract fixations:")
    use_parallel = params.get('use_parallel', True)
    all_fix_timepos, fix_detection_results = extract_or_load_fixations(
        labelled_gaze_positions, params)
    labelled_fixations = generate_fixation_labels(
        fix_detection_results, params, use_parallel)
    return labelled_fixations


def extract_or_load_fixations(labelled_gaze_positions, params):
    """
    Extract or load fixations based on parameters.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions
    and associated metadata.
    - params (dict): Dictionary of parameters.
    Returns:
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
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    """
    all_fix_timepos_df = load_data.load_m1_fixations(params)
    fix_detection_results = load_data.load_fix_detection_results(params)
    return all_fix_timepos_df, fix_detection_results


def generate_fixation_labels(fix_detection_results, params, use_parallel):
    """
    Generate fixation labels based on detection results.
    Parameters:
    - fix_detection_results (list): List of fixation detection results.
    - params (dict): Dictionary of parameters.
    - use_parallel (bool): Whether to use parallel processing.
    Returns:
    - labelled_fixations (pd.DataFrame): DataFrame of labels for fixations.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    if params.get('remake_fixation_labels', False):
        fixation_labels = parallel_generate_labels(
            fix_detection_results) if use_parallel \
            else serial_generate_labels(fix_detection_results)
        labelled_fixations = []
        for session_labels in fixation_labels:
            labelled_fixations.extend(session_labels)
        col_names = ['start_time', 'end_time', 'category', 'session_name',
                     'run', 'block', 'fix_duration', 'mean_x_pos',
                     'mean_y_pos', 'fix_roi', 'agent']
        labelled_fixations = pd.DataFrame(labelled_fixations,
                                          columns=col_names)
        fixation_labels_file_name = f'fixation_labels_m1{flag_info}.csv'
        labelled_fixations.to_csv(os.path.join(
            processed_data_dir,fixation_labels_file_name), index=False)
    else:
        labelled_fixations = load_data.load_m1_fixation_labels(params)
    return labelled_fixations


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
            add_labels_to_fixations, fix_detection_results)
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
    return [add_labels_to_fixations(result)
            for result in fix_detection_results]


def extract_all_fixations_from_labelled_gaze_positions(labelled_gaze_positions, params):
    """
    Extracts fixations from labelled gaze positions.
    Parameters:
    - labelled_gaze_positions (list): List of labelled gaze positions.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    """
    processed_data_dir = params['processed_data_dir']
    use_parallel = params.get('use_parallel', True)
    sessions_data = [(session_data[0], session_data[1], params)
                     for session_data in labelled_gaze_positions]
    fix_detection_results = extract_fixations(sessions_data, use_parallel)
    all_fix_timepos = process_fixation_results(fix_detection_results)
    save_fixation_results(processed_data_dir, all_fix_timepos, params)
    return all_fix_timepos, fix_detection_results


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
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    """
    all_fix_timepos = pd.DataFrame()
    for session_timepos_df, _ in fix_detection_results:
        all_fix_timepos = pd.concat(
            [all_fix_timepos, session_timepos_df], ignore_index=True)
    return all_fix_timepos


def save_fixation_results(processed_data_dir, all_fix_timepos, params):
    """
    Saves fixation results to files.
    Parameters:
    - processed_data_dir (str): Directory to save processed data.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - params (dict): Dictionary of parameters.
    """
    flag_info = util.get_filename_flag_info(params)
    timepos_file_name = f'fix_timepos_m1{flag_info}.csv'
    all_fix_timepos.to_csv(os.path.join(
        processed_data_dir, timepos_file_name), index=False)


def get_session_fixations(session_data):
    """
    Extracts fixations for a session.
    Parameters:
    - session_data (tuple): Tuple containing session identifier, positions,
    and metadata.
    Returns:
    - fix_timepos_df (pd.DataFrame): DataFrame of fixation time positions.
    - info (dict): Metadata information for the session.
    """
    positions, info, params = session_data
    session_name = info['session_name']
    sampling_rate = info['sampling_rate']
    n_samples = positions.shape[0]
    time_vec = util.create_timevec(n_samples, sampling_rate)
    fix_timepos_df, fix_vec_entire_session = fix.is_fixation(
        positions, time_vec, session_name, sampling_rate=sampling_rate)
    return fix_timepos_df, info


### Function to generate fixation labels for each session
def add_labels_to_fixations(fix_detection_result):
    """
    Generates fixation labels for each session.
    Parameters:
    - fix_detection_result (tuple): Tuple containing fixation detection results
    and session information.
    Returns:
    - fixation_labels (list): List of fixation labels.
    """
    session_timepos_df, info = fix_detection_result
    fixation_labels = []
    for _, row in tqdm(session_timepos_df.iterrows(), desc=f"{info['session_name']}: n fixations labelled"):
        fix_x = row['fix_x']
        fix_y = row['fix_y']
        start_time = row['start_time']
        end_time = row['end_time']
        fix_duration = row['duration']
        mean_fix_pos = [fix_x, fix_y]
        run, block, fix_roi = detect_run_block_and_roi(
            [start_time, end_time], info['startS'], info['stopS'],
            info['sampling_rate'], mean_fix_pos, info['roi_bb_corners'])
        fixation_info = [start_time, end_time, info['category'],
                         info['session_name'], run, block, fix_duration,
                         mean_fix_pos[0], mean_fix_pos[1], fix_roi,
                         info['monkey_1']]
        fixation_labels.append(fixation_info)
    return fixation_labels


### Function to detect run, block, and ROI for a fixation
def detect_run_block_and_roi(start_stop, startS, stopS, 
                            sampling_rate, mean_fix_pos, bbox_corners):
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
    # Check if fixation is before the first start time or after the last stop time
    if start < startS[0] or stop > stopS[-1]:
        run = None
        block = 'discard'
    else:
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
    bounding_boxes = ['eye_bbox', 'face_bbox',
                      'left_obj_bbox', 'right_obj_bbox']
    inside_roi = [util.is_inside_roi(mean_fix_pos, bbox_corners[key])
                  for key in bounding_boxes]
    if any(inside_roi):
        if inside_roi[0] and inside_roi[1]:
            return bounding_boxes[0]
        return bounding_boxes[inside_roi.index(True)]
    return 'out_of_roi'


### Function to extract saccades with labels
def extract_saccades_with_labels(labelled_gaze_positions, params):
    """
    Extracts saccades with labels.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    - params (dict): Dictionary containing parameters including parallel processing options.
    Returns:
    - labelled_saccades (DataFrame): DataFrame containing saccade information with labels.
    """
    saccade_params = defaults.fetch_default_saccade_pars()
    vel_thresh = saccade_params['vel_thresh']
    min_samples = saccade_params['min_samples']
    smooth_func = saccade_params['smooth_func']
    use_parallel = params.get('use_parallel', False)
    num_sessions = len(labelled_gaze_positions)
    available_cpus = os.cpu_count()
    n_jobs = min(available_cpus, num_sessions)
    sessions_data = [(session_data[0], session_data[1],
                      vel_thresh, min_samples, smooth_func)
                     for session_data in labelled_gaze_positions]
    if use_parallel:
        with tqdm_joblib(tqdm(
                desc="Extracting saccades in parallel",
                total=num_sessions, unit="session")):
            results = Parallel(n_jobs=n_jobs)(
                delayed(extract_saccades_for_session)(session_data)
                for session_data in sessions_data)
    else:
        results = [extract_saccades_for_session(session_data)
                   for session_data in
                   tqdm(sessions_data,
                        desc="Extracting saccades",
                        unit="session")]
    saccades = [s for session_saccades in results for s in session_saccades]
    columns = ["start_time", "end_time", "duration", "trajectory",
               "start_roi", "end_roi", "session_name", "category",
               "run", "block"]
    labelled_saccades = pd.DataFrame(saccades, columns=columns)
    save_saccade_labels(labelled_saccades, params)
    return labelled_saccades


def extract_saccades_for_session(session_data):
    """
    Extracts saccades for a single session.
    Parameters:
    - session_data (tuple): Tuple containing gaze positions, session info, and saccade parameters.
    Returns:
    - session_saccades (list): List of saccades for the session.
    """
    positions, info, vel_thresh, min_samples, smooth_func = session_data
    session_saccades = []
    sampling_rate = info['sampling_rate']
    n_samples = positions.shape[0]
    time_vec = util.create_timevec(n_samples, sampling_rate)
    category = info['category']
    session_name = info['session_name']
    n_runs = info['num_runs']
    for run in range(n_runs):
        run_start = info['startS'][run]
        run_stop = info['stopS'][run]
        run_time = (time_vec > run_start) & (time_vec <= run_stop)
        run_positions = positions[run_time, :]
        run_x = util.px2deg(run_positions[:, 0].T)
        run_y = util.px2deg(run_positions[:, 1].T)
        saccade_start_stops = find_saccades(
            run_x, run_y, sampling_rate, vel_thresh,
            min_samples, smooth_func)
        for start, stop in saccade_start_stops:
            saccade = run_positions[start:stop + 1, :]
            start_time = time_vec[start]
            end_time = time_vec[stop]
            duration = end_time - start_time
            start_roi = determine_roi_of_coord(run_positions[start, :2],
                                      info['roi_bb_corners'])
            end_roi = determine_roi_of_coord(run_positions[stop, :2],
                                    info['roi_bb_corners'])
            block = determine_block(
                start_time, end_time, info['startS'], info['stopS'])
            session_saccades.append(
                [start_time, end_time, duration, saccade,
                 start_roi, end_roi, session_name, category, run, block])
    return session_saccades


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
    - start_stops (list): List of start and stop indices of saccades.
    """
    assert x.shape == y.shape
    start_stops = []
    x0 = smooth_func(x)
    y0 = smooth_func(y)
    vx = np.gradient(x0) / sr
    vy = np.gradient(y0) / sr
    vel_norm = np.sqrt(vx ** 2 + vy ** 2)  # Norm of velocity vector
    above_thresh = (vel_norm >= vel_thresh[0]) & (vel_norm <= vel_thresh[1])
    start_stops = util.find_islands(above_thresh, min_samples)
    return start_stops


def determine_roi_of_coord(position, bbox_corners):
    """
    Determines the ROI based on position and bounding box corners.
    Parameters:
    - position (ndarray): Position coordinates.
    - bbox_corners (dict): Dictionary containing bounding boxes of ROIs.
    Returns:
    - roi (str): Detected ROI.
    """
    bounding_boxes = ['eye_bbox', 'face_bbox',
                      'left_obj_bbox', 'right_obj_bbox']
    inside_roi = [util.is_inside_roi(position, bbox_corners[key])
                  for key in bounding_boxes]
    if any(inside_roi):
        if inside_roi[0] and inside_roi[1]:
            return bounding_boxes[0]
        return bounding_boxes[inside_roi.index(True)]
    return 'out_of_roi'


def determine_block(start_time, end_time, startS, stopS):
    """
    Determines the block for a saccade based on start and stop times.
    Parameters:
    - start_time (float): Start time of the saccade.
    - end_time (float): End time of the saccade.
    - startS (list): List of start indices of runs.
    - stopS (list): List of stop indices of runs.
    Returns:
    - block (str): Detected block.
    """
    if start_time < startS[0] or end_time > stopS[-1]:
        return 'discard'
    for i, (run_start, run_stop) in enumerate(zip(startS, stopS), start=1):
        if start_time >= run_start and end_time <= run_stop:
            return 'mon_down'
        elif i < len(startS) and end_time <= startS[i]:
            return 'mon_up'
    return 'discard'


def save_saccade_labels(labelled_saccades, params):
    """
    Saves the labelled saccades to a specified directory.
    Parameters:
    - labelled_saccades (DataFrame): DataFrame containing saccade information with labels.
    - params (dict): Dictionary containing parameters including the save directory.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    file_path = os.path.join(
        processed_data_dir, f'labelled_saccades{flag_info}.csv')
    labelled_saccades.to_csv(file_path, index=False)
    print(f"Saccade labels saved to {file_path}")


def extract_spiketimes_for_all_sessions(params):
    processed_data_dir = params.get('processed_data_dir')
    session_paths = params.get('session_paths')
    is_parallel = params.get('use_parallel', True)
    spikeTs_labels = []
    if is_parallel:
        # Process sessions in parallel with tqdm progress bar
        results = Parallel(n_jobs=-1)(delayed(
            load_data.get_spiketimes_and_labels_for_one_session)(
                session_path, processed_data_dir)
            for session_path in tqdm(
                    session_paths, desc='Loading spiketimes'))
        # Iterate over results and concatenate
        for labelled_spiketimes in tqdm(
                results, desc='concatenating results'):
            spikeTs_labels.append(labelled_spiketimes)
    else:
        # Process sessions sequentially
        for session_path in session_paths:
            labelled_spiketimes = \
                load_data.get_spiketimes_and_labels_for_one_session(
                    session_path, processed_data_dir)
            spikeTs_labels.append(labelled_spiketimes)
    # Concatenate label dataframes
    if spikeTs_labels:
        all_labels = pd.concat(spikeTs_labels, ignore_index=True)
    else:
        all_labels = pd.DataFrame()
    # Construct flag_info based on params
    flag_info = util.get_filename_flag_info(params)
    # Save outputs to root_data_dir with flag_info
    labels_path = os.path.join(
        processed_data_dir, f'spike_labels{flag_info}.csv')
    # Save DataFrame
    all_labels.to_csv(labels_path, index=False)
    print(f"All labelled spiketimes saved to {labels_path}")
    return all_labels


def extract_fixation_raster(labelled_fixations, labelled_spiketimes, params):
    """
    Main function to generate rasters for all sessions.
    Parameters:
    labelled_fixations (pd.DataFrame): DataFrame containing fixation data.
    labelled_spiketimes (pd.DataFrame): DataFrame containing spiketimes data.
    params (dict): Dictionary containing parameters for raster generation.
    Returns:
    pd.DataFrame: DataFrame containing all generated rasters and labels.
    """
    sessions = labelled_fixations['session_name'].unique()
    results = []
    num_sessions = len(sessions)
    num_cores = os.cpu_count()
    num_processes = min(num_sessions, num_cores)
    if params.get('use_parallel', False):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(
                generate_session_raster, session, labelled_fixations,
                labelled_spiketimes, params): session for session in sessions}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Generating raster for session"):
                results.append(future.result())
    else:
        for session in tqdm(sessions, desc="Generating raster for session"):
            results.append(generate_session_raster(
                session, labelled_fixations, labelled_spiketimes, params))
    labelled_fixation_rasters = pd.concat(results, ignore_index=True)
    save_labelled_fixation_rasters(labelled_fixation_rasters, params)
    return labelled_fixation_rasters



def generate_session_raster(session, labelled_fixations, labelled_spiketimes, params):
    """
    Function to generate rasters for a single session.
    Parameters:
    session (str): The name of the session to process.
    labelled_fixations (pd.DataFrame): DataFrame containing fixation data.
    labelled_spiketimes (pd.DataFrame): DataFrame containing spiketimes data.
    params (dict): Dictionary containing parameters for raster generation.
    Returns:
    pd.DataFrame: DataFrame containing rasters and labels for the session.
    """
    # Extract parameters
    raster_bin_size = float(params['raster_bin_size'])
    raster_pre_event_time = float(params['raster_pre_event_time'])
    raster_post_event_time = float(params['raster_post_event_time'])
    num_bins = int(
        (raster_pre_event_time + raster_post_event_time) / raster_bin_size)
    # Filter data for the current session
    session_fixations = labelled_fixations[
        labelled_fixations['session_name'] == session]
    session_neurons = labelled_spiketimes[
        labelled_spiketimes['session_name'] == session]
    # Pre-allocate memory for the dataframe
    num_neurons = session_neurons['uuid'].nunique()
    num_fixations = session_fixations.shape[0]
    num_rasters = num_neurons * num_fixations * 2  # 2 for start_time and end_time
    # Pre-initialize the DataFrame
    columns = ['raster', 'category', 'session_name', 'run', 'block',
               'fix_duration', 'mean_x_pos', 'mean_y_pos', 'fix_roi',
               'agent', 'channel', 'channel_label', 'unit_no_within_channel',
               'unit_label', 'uuid', 'n_spikes', 'region',
               'aligned_to', 'behavior']
    session_data = pd.DataFrame(index=np.arange(num_rasters), columns=columns)
    idx = 0  # Index for pre-allocated dataframe
    # Create a tqdm progress bar for unit processing within the session
    for uuid in tqdm(session_neurons['uuid'].unique(),
                     desc=f"Processing unit in session {session}"):
        neuron_spikes_str = session_neurons[
            session_neurons['uuid'] == uuid]['spikeS'].values[0]
        neuron_spikes = np.array(ast.literal_eval(neuron_spikes_str))
        for _, fixation in session_fixations.iterrows():
            for aligned_to in ['start_time', 'end_time']:
                event_time = float(fixation[aligned_to])
                window_start = event_time - raster_pre_event_time
                window_end = event_time + raster_post_event_time
                # Filter spikes within the window of interest
                relevant_spikes = neuron_spikes[
                    (neuron_spikes >= window_start) &
                    (neuron_spikes < window_end)]
                # Initialize binary raster
                raster = np.zeros(num_bins, dtype=int)
                # Update the raster with relevant spikes
                for spike_time in relevant_spikes:
                    bin_idx = int(
                        (spike_time - window_start) / raster_bin_size)
                    if bin_idx < num_bins:
                        raster[bin_idx] = 1
                session_data = update_session_data(session_data, idx, raster,
                                                   fixation, session_neurons,
                                                   uuid, aligned_to)
                idx += 1
    return session_data


def generate_binary_raster(neuron_spikes, bins):
    """
    Function to generate a binary raster for the given neuron spikes and bins.
    Parameters:
    neuron_spikes (np.ndarray): Array containing neuron spike times.
    bins (np.ndarray): Array containing the bin edges.
    Returns:
    np.ndarray: Binary raster.
    """
    raster = np.histogram(neuron_spikes, bins=bins)[0]
    raster = (raster > 0).astype(int)
    return raster


def update_session_data(session_data, idx, raster, fixation, session_neurons, uuid, aligned_to):
    """
    Function to update the session data DataFrame with raster and label information.
    Parameters:
    session_data (pd.DataFrame): DataFrame to be updated.
    idx (int): Index to update.
    raster (np.ndarray): Binary raster.
    fixation (pd.Series): Series containing fixation information.
    session_neurons (pd.DataFrame): DataFrame containing neuron information.
    uuid (str): Unique identifier for the neuron.
    aligned_to (str): Indicates whether the raster is aligned to start_time or end_time.
    Returns:
    pd.DataFrame: Updated session data DataFrame.
    """
    session_data.at[idx, 'raster'] = raster
    session_data.at[idx, 'category'] = fixation['category']
    session_data.at[idx, 'session_name'] = fixation['session_name']
    session_data.at[idx, 'run'] = fixation['run']
    session_data.at[idx, 'block'] = fixation['block']
    session_data.at[idx, 'fix_duration'] = fixation['fix_duration']
    session_data.at[idx, 'mean_x_pos'] = fixation['mean_x_pos']
    session_data.at[idx, 'mean_y_pos'] = fixation['mean_y_pos']
    session_data.at[idx, 'fix_roi'] = fixation['fix_roi']
    session_data.at[idx, 'agent'] = fixation['agent']
    session_data.at[idx, 'channel'] = session_neurons[
        session_neurons['uuid'] == uuid]['channel'].values[0]
    session_data.at[idx, 'channel_label'] = session_neurons[
        session_neurons['uuid'] == uuid]['channel_label'].values[0]
    session_data.at[idx, 'unit_no_within_channel'] = session_neurons[
        session_neurons['uuid'] == uuid]['unit_no_within_channel'].values[0]
    session_data.at[idx, 'unit_label'] = session_neurons[
        session_neurons['uuid'] == uuid]['unit_label'].values[0]
    session_data.at[idx, 'uuid'] = uuid
    session_data.at[idx, 'n_spikes'] = session_neurons[
        session_neurons['uuid'] == uuid]['n_spikes'].values[0]
    session_data.at[idx, 'region'] = session_neurons[
        session_neurons['uuid'] == uuid]['region'].values[0]
    session_data.at[idx, 'aligned_to'] = aligned_to
    session_data.at[idx, 'behavior'] = 'fixation'
    return session_data


def save_labelled_fixation_rasters(labelled_fixation_rasters, params):
    """
    Function to save the labelled fixation rasters DataFrame to a specified directory.

    Parameters:
    labelled_fixation_rasters (pd.DataFrame): DataFrame containing all generated rasters and labels.
    params (dict): Dictionary containing parameters including the directory to save the processed data.
    flag_info (str): Additional flag information to append to the filename.
    """
    processed_data_dir = params['processed_data_dir']
    # Create directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    # Construct the filename
    flag_info = util.get_filename_flag_info(params)
    filename = f"labelled_fixation_rasters{flag_info}.csv"
    file_path = os.path.join(processed_data_dir, filename)
    # Save the DataFrame to a CSV file
    labelled_fixation_rasters.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

















