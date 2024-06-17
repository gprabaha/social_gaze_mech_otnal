#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:34:44 2024

@author: pg496
"""
"""
Script for fixation detection
"""

import numpy as np
import util  # Import utility functions here
from tqdm import tqdm
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool

import load_data

import pdb  # Import the Python debugger if needed


###
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


###
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
    fix_timepos_df, fix_vec_entire_session = is_fixation(
        positions, time_vec, session_name, sampling_rate=sampling_rate)
    return fix_timepos_df, info


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


def is_fixation(pos, time, session_name, t1=None, t2=None, minDur=None, maxDur=None, sampling_rate=None):
    """
    Determine fixations based on position and time data.
    Args:
    pos: Position data (x, y).
    time: Time data.
    t1: Spatial parameter t1.
    t2: Spatial parameter t2.
    minDur: Minimum fixation duration.
    sampling_rate: Sampling rate.
    Returns:
    Binary vector indicating fixations (1) and non-fixations (0).
    """
    # Combine position and time into a single data matrix
    data = np.column_stack((pos, time))
    # Calculate sampling rate if not provided
    if sampling_rate is None:
        sampling_rate = 1 / (time[1,:] - time[0,:])
    # Set default values for parameters if not provided
    if minDur is None:
        minDur = 0.05
    if maxDur is None:
        maxDur = 2
    if t2 is None:
        t2 = 15
    if t1 is None:
        t1 = 30
    # Initialize fix_vector
    fix_vector = np.zeros(data.shape[0])
    '''
    Implement a proper outlier detection code here. The curve fit should
    account for a situation where t_n is (x_a, y_a) and t_m is (x_a, y_b), so
    the same x will have 2 y values, which can get complicated for polyfit
    '''
    fix_list_df, fix_t_inds = fixation_detection(data, t1, t2, minDur, maxDur, session_name)
    for t_range in fix_t_inds:
        fix_vector[t_range[0]:t_range[1] + 1] = 1
    return fix_list_df, fix_vector


###
def fixation_detection(data, t1, t2, minDur, maxDur, session_name):
    """
    Detect fixations based on position and time data.
    Args:
    data: Combined position and time data.
    t1: Spatial parameter t1.
    t2: Spatial parameter t2.
    minDur: Minimum fixation duration.
    Returns:
    List of fixation time ranges.
    """
    n = len(data)
    if n == 0:
        return []  # Return empty list if data is empty
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    n = len(t)
    fixations = get_t1_filtered_fixations(n, x, y, t, t1, session_name)
    number_fixations = fixations[-1, 3]
    fixation_list = []
    for i in tqdm(range(1, int(number_fixations) + 1),
                  desc=f"{session_name}: n fixations t2 filtered"):
        fixation_list.append(filter_fixations_t2(i, fixations, t2))
    # Duration thresholding
    fixation_list = min_duration(fixation_list, minDur)
    fixation_list = max_duration(fixation_list, maxDur)
    # Convert fixation list to time ranges
    fix_ranges = []
    for fix in fixation_list:
        s_ind = np.where(data[:, 2] == fix[4])[0][0]
        e_ind = np.where(data[:, 2] == fix[5])[0][-1]
        fix_ranges.append([s_ind, e_ind])
    col_names = ['fix_x', 'fix_y', 'threshold_1', 'threshold_2',
                 'start_time', 'end_time', 'duration']
    return pd.DataFrame(fixation_list, columns=col_names), fix_ranges


def get_t1_filtered_fixations(n, x, y, t, t1, session_name):
    """
    Filter fixations based on spatial parameter t1.
    Args:
    n: Length of data.
    x: X-coordinate data.
    y: Y-coordinate data.
    t: Time data.
    t1: Spatial parameter t1.
    Returns:
    Array of fixations after filtering.
    """
    fixations = np.zeros((n, 4))
    fixid = 0
    fixpointer = 0
    for i in tqdm(range(n),
                  desc='{}: n data points t1 filtered'.format(session_name)):
        if not np.any(x[fixpointer:i+1]) or not np.any(y[fixpointer:i+1]):
            fixations = update_fixations(i, x, y, t, fixations, fixid)
        else:
            mx = np.mean(x[fixpointer:i+1])
            my = np.mean(y[fixpointer:i+1])
            d = util.distance2p(mx, my, x[i], y[i])
            if d > t1:
                fixid += 1
                fixpointer = i
            fixations = update_fixations(i, x, y, t, fixations, fixid)
    return fixations


def update_fixations(i, x, y, t, fixations, fixid):
    """
    Update fixations array with new fixation data.
    Args:
    i: Index.
    x: X-coordinate data.
    y: Y-coordinate data.
    t: Time data.
    fixations: Array of fixations.
    fixid: ID of the fixation.
    Returns:
    Updated fixations array.
    """
    fixations[i, 0] = x[i]
    fixations[i, 1] = y[i]
    fixations[i, 2] = t[i]
    fixations[i, 3] = fixid
    return fixations


def filter_fixations_t2(fixation_id, fixations, t2):
    """
    Cluster fixations based on spatial criteria and apply t2 threshold.
    Args:
    fixation_id: ID of the fixation.
    fixations: Array containing fixations.
    t2: Spatial parameter t2.
    Returns:
    Fixation information after applying t2 threshold.
    """
    fixations_id = fixations[fixations[:, 3] == fixation_id]
    number_t1 = len(fixations_id)
    # Clustering according to criterion t2
    fixx, fixy = np.nanmean(fixations_id[:, :2], axis=0)
    for i in range(number_t1):
        d = util.distance2p(fixx, fixy, fixations_id[i, 0], fixations_id[i, 1])
        if d > t2:
            fixations_id[i, 3] = 0
    # Initialize lists
    fixations_list_t2 = np.empty((0, 4))
    list_out_points = np.empty((0, 4))
    for i in range(number_t1):
        if fixations_id[i, 3] > 0:
            fixations_list_t2 = np.vstack((fixations_list_t2, fixations_id[i, :]))
        else:
            list_out_points = np.vstack((list_out_points, fixations_id[i, :]))
    # Compute number of t2 fixations
    number_t2 = fixations_list_t2.shape[0]
    if not np.any(fixations_list_t2[:, :2]):
        start_time, end_time, duration = 0, 0, 0
    else:
        fixx, fixy = np.nanmean(fixations_list_t2[:, :2], axis=0)
        start_time = fixations_list_t2[0, 2]
        end_time = fixations_list_t2[-1, 2]
        duration = end_time - start_time
    return fixx, fixy, number_t1, number_t2, start_time, end_time, duration


def min_duration(fixation_list, minDur):
    """
    Apply duration threshold to fixation list.
    Args:
    fixation_list: List of fixations.
    minDur: Minimum fixation duration.
    Returns:
    Fixation list after applying duration threshold.
    """
    return [fix for fix in fixation_list if fix[6] >= minDur]

def max_duration(fixation_list, maxDur):
    return [fix for fix in fixation_list if fix[6] <= maxDur]


###
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


###
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