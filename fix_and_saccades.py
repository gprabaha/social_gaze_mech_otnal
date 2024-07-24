#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:34:44 2024

@author: pg496
"""


import numpy as np
import util  # Import utility functions here
import pickle
import pandas as pd
import os
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import load_data
from cluster_fix import ClusterFixationDetector  # Import the new ClusterFixationDetector class
from eye_mvm_fix import EyeMVMFixationDetector  # Import the new EyeMVMFixationDetector class
from eye_mvm_saccade import EyeMVMSaccadeDetector  # Import the new EyeMVMSaccadeDetector class
from hpc_fixation_detection import HPCFixationDetection  # Import the new HPCFixationDetection class

import pdb


def extract_or_load_fixations_and_saccades(labelled_gaze_positions, params):
    """
    Extract or load fixations and saccades based on parameters.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    - saccade_detection_results (list): List of saccade detection results.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    if params.get('remake_labelled_fixations', False) or params.get('remake_labelled_saccades_m1', False):
        return extract_all_fixations_and_saccades_from_labelled_gaze_positions(labelled_gaze_positions, params)
    else:
        results_file_name = f'fixation_saccade_session_results_m1{flag_info}.npz'
        if os.path.exists(os.path.join(processed_data_dir, results_file_name)):
            return load_existing_fixations_and_saccades(params)
        else:
            return extract_all_fixations_and_saccades_from_labelled_gaze_positions(labelled_gaze_positions, params)


def load_existing_fixations_and_saccades(params):
    """
    Load existing fixations and saccades from files.
    Parameters:
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    - saccade_detection_results (list): List of saccade detection results.
    """
    all_fix_timepos_df = load_data.load_m1_fixations(params)
    fix_detection_results = load_data.load_fix_detection_results(params)
    saccade_detection_results = load_data.load_saccade_detection_results(params)
    return all_fix_timepos_df, fix_detection_results, saccade_detection_results



def extract_all_fixations_and_saccades_from_labelled_gaze_positions(labelled_gaze_positions, params):
    """
    Extracts fixations and saccades from labelled gaze positions.
    Parameters:
    - labelled_gaze_positions (list): List of labelled gaze positions.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fix_df (pd.DataFrame): DataFrame of all fixation time positions.
    - all_saccades_df (pd.DataFrame): DataFrame of all saccades.
    """
    processed_data_dir = params['processed_data_dir']
    use_parallel = params.get('use_parallel', True)
    submit_separate_jobs = params.get('submit_separate_jobs_for_sessions', True)
    if submit_separate_jobs:
        hpc_fixation_detection = HPCFixationDetection(params)
        job_file_path = hpc_fixation_detection.generate_fixation_job_file(labelled_gaze_positions)
        hpc_fixation_detection.submit_job_array(job_file_path)
        session_files = [os.path.join(processed_data_dir, f"{i}_fixations.pkl") for i in range(len(labelled_gaze_positions))]
        results = []
        for session_file in session_files:
            try:
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                results.append(session_data)
            except FileNotFoundError as e:
                logging.error(e)
                continue
        if not results:
            logging.error("No results to concatenate.")
            raise ValueError("No objects to concatenate")
        all_fix_df, all_info, all_saccades_df = zip(*results)
    else:
        sessions_data = [(session_data[0], session_data[1], params) for session_data in labelled_gaze_positions]
        num_cpus = params['num_cpus']
        all_fix_df, all_saccades_df = extract_fixations_and_saccades(sessions_data, use_parallel, num_cpus)
    all_fix_df, all_saccades_df = process_detection_results(all_fix_df, all_saccades_df)
    save_fixation_and_saccade_results(processed_data_dir, all_fix_df, all_saccades_df, params)
    return all_fix_df, all_saccades_df


def process_detection_results(fix_detection_results, saccade_detection_results):
    """
    Processes the results from fixation and saccade detection.
    Parameters:
    - fix_detection_results (list): List of fixation detection results.
    - saccade_detection_results (list): List of saccade detection results.
    Returns:
    - all_fix_timepos (pd.DataFrame): DataFrame of all fixation time positions.
    - all_saccades (pd.DataFrame): DataFrame of all saccades.
    """
    all_fix_timepos = pd.DataFrame()
    all_saccades = pd.DataFrame()
    for fix_df, saccade_df in zip(fix_detection_results, saccade_detection_results):
        all_fix_timepos = pd.concat([all_fix_timepos, fix_df], ignore_index=True)
        all_saccades = pd.concat([all_saccades, saccade_df], ignore_index=True)
    return all_fix_timepos, all_saccades


def extract_fixations_and_saccades(sessions_data, use_parallel, num_cpus):
    """
    Extracts fixations and saccades from session data.
    Parameters:
    - sessions_data (list): List of session data tuples.
    - use_parallel (bool): Flag to determine if parallel processing should be used.
    - num_cpus (int): Number of CPUs to use for parallel processing.
    Returns:
    - fix_detection_results (list): List of fixation detection results.
    - saccade_detection_results (list): List of saccade detection results.
    """
    if use_parallel:
        print("\nExtracting fixations and saccades in parallel")
        num_processes = min(num_cpus, len(sessions_data))
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(get_session_fixations_and_saccades, session_data, num_cpus): session_data for session_data in sessions_data}
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        print("\nExtracting fixations and saccades serially")
        results = [get_session_fixations_and_saccades(session_data, num_cpus) for session_data in sessions_data]
    # Unpack the results into separate lists for fixations and saccades
    fix_detection_results = []
    saccade_detection_results = []
    for result in results:
        fix_timepos_df, info, saccades_df = result
        fix_detection_results.append(fix_timepos_df)
        saccade_detection_results.append(saccades_df)
    return fix_detection_results, saccade_detection_results



def get_session_fixations_and_saccades(session_data, num_cpus):
    """
    Extracts fixations and saccades for a session.
    Parameters:
    - session_data (tuple): Tuple containing session identifier, positions, and metadata.
    Returns:
    - fix_timepos_df (pd.DataFrame): DataFrame of fixation time positions.
    - info (dict): Metadata information for the session.
    - saccades_df (pd.DataFrame): DataFrame of saccades for the session.
    """
    positions, info, params = session_data
    session_name = info['session_name']
    sampling_rate = info['sampling_rate']
    n_samples = positions.shape[0]
    time_vec = util.create_timevec(n_samples, sampling_rate)
    use_parallel = params.get('use_parallel', False)
    detector = ClusterFixationDetector(samprate=sampling_rate, params=params, num_cpus=num_cpus)
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    # Transform into the expected format
    eyedat = (x_coords, y_coords)
    fix_stats = detector.detect_fixations(eyedat)
    fixations_df = make_fixations_df(fix_stats, info)
    saccades_df = make_saccades_df(fix_stats, info)
    print(fixations_df.head())
    print(saccades_df.head())
    return fixations_df, info, saccades_df



def make_fixations_df(fix_stats, info):
    """
    Creates a DataFrame for fixations.
    Parameters:
    - fix_stats (dict): Dictionary containing fixation statistics.
    - info (dict): Dictionary of session information.
    Returns:
    - fix_timepos_df (pd.DataFrame): DataFrame of fixation time positions.
    """
    fixationindices = fix_stats['fixationindices']
    fixationtimes = fix_stats['fixationtimes']
    positions = fix_stats['XY'].T
    # Calculate points in events
    points_in_event = [positions[start:end+1, :] for start, end in zip(fixationindices[0, :], fixationindices[1, :])]
    # Calculate mean position and std spread for each event
    mean_position = [np.nanmean(point, axis=0) for point in points_in_event]
    std_spread = [np.nanstd(point, axis=0) for point in points_in_event]
    fix_timepos_df = pd.DataFrame({
        'start_index': fixationindices[0, :].T,
        'end_index': fixationindices[1, :].T,
        'start_time': fixationtimes[0, :].T,
        'end_time': fixationtimes[1, :].T,
        'points_in_event': points_in_event,
        'mean_position': mean_position,
        'std_spread': std_spread
    })
    # Calculate additional fixation parameters
    fix_timepos_df['duration'] = fix_timepos_df['end_time'] - fix_timepos_df['start_time']
    fix_timepos_df['start_roi'] = fix_timepos_df.apply(lambda row: determine_roi_of_coord(row['mean_position'], info['roi_bb_corners']), axis=1)
    fix_timepos_df['end_roi'] = fix_timepos_df['start_roi']
    fix_timepos_df['category'] = info['category']
    fix_timepos_df['session_name'] = info['session_name']
    fix_timepos_df['run'] = fix_timepos_df.apply(
        lambda row: determine_run(row['start_time'], row['end_time'], info['startS'], info['stopS']),
        axis=1
    )
    fix_timepos_df['block'] = fix_timepos_df.apply(lambda row: determine_block(row['start_time'], row['end_time'], info['startS'], info['stopS']), axis=1)
    fix_timepos_df['agent'] = info.get('monkey_1', None)
    fix_timepos_df['partner'] = info.get('monkey_2', None)
    # Reorder columns
    fix_timepos_df = fix_timepos_df[[
        'session_name', 'run', 'block', 'agent', 'partner', 
        'start_index', 'end_index', 'start_time', 'end_time', 
        'duration', 'points_in_event', 'mean_position', 
        'std_spread', 'start_roi', 'end_roi', 'category'
    ]]
    return fix_timepos_df


def make_saccades_df(saccade_stats, info):
    """
    Creates a DataFrame for saccades.
    Parameters:
    - saccade_stats (dict): Dictionary containing saccade statistics.
    - info (dict): Dictionary of session information.
    Returns:
    - saccade_timepos_df (pd.DataFrame): DataFrame of saccade time positions.
    """
    saccadeindices = saccade_stats['saccadeindices']
    saccadetimes = saccade_stats['saccadetimes']
    positions = saccade_stats['XY'].T
    # Calculate points in events
    points_in_event = [positions[start:end+1, :] for start, end in zip(saccadeindices[0, :], saccadeindices[1, :])]
    # Calculate mean position and std spread for each event
    mean_position = [np.nanmean(point, axis=0) for point in points_in_event]
    std_spread = [np.nanstd(point, axis=0) for point in points_in_event]
    saccade_timepos_df = pd.DataFrame({
        'start_index': saccadeindices[0, :].T,
        'end_index': saccadeindices[1, :].T,
        'start_time': saccadetimes[0, :].T,
        'end_time': saccadetimes[1, :].T,
        'points_in_event': points_in_event,
        'mean_position': mean_position,
        'std_spread': std_spread
    })
    # Calculate additional saccade parameters
    saccade_timepos_df['duration'] = saccade_timepos_df['end_time'] - saccade_timepos_df['start_time']
    saccade_timepos_df['start_roi'] = saccade_timepos_df.apply(lambda row: determine_roi_of_coord(row['mean_position'], info['roi_bb_corners']), axis=1)
    saccade_timepos_df['end_roi'] = saccade_timepos_df['start_roi']
    saccade_timepos_df['category'] = info['category']
    saccade_timepos_df['session_name'] = info['session_name']
    saccade_timepos_df['run'] = saccade_timepos_df.apply(
        lambda row: determine_run(row['start_time'], row['end_time'], info['startS'], info['stopS']),
        axis=1
    )
    saccade_timepos_df['block'] = saccade_timepos_df.apply(lambda row: determine_block(row['start_time'], row['end_time'], info['startS'], info['stopS']), axis=1)
    saccade_timepos_df['agent'] = info.get('monkey_1', None)
    saccade_timepos_df['partner'] = info.get('monkey_2', None)
    # Reorder columns
    saccade_timepos_df = saccade_timepos_df[[
        'session_name', 'run', 'block', 'agent', 'partner', 
        'start_index', 'end_index', 'start_time', 'end_time', 
        'duration', 'points_in_event', 'mean_position', 
        'std_spread', 'start_roi', 'end_roi', 'category'
    ]]
    return saccade_timepos_df



# Determine the run number based on start and end times
def determine_run(start_time, end_time, startS, stopS):
    for run_number, (start, stop) in enumerate(zip(startS, stopS), start=1):
        if start <= end_time and stop >= start_time:
            return run_number
    return None


def determine_roi_of_coord(position, bbox_corners):
    bounding_boxes = ['eye_bbox', 'face_bbox', 'left_obj_bbox', 'right_obj_bbox']
    inside_roi = [util.is_inside_roi(position, bbox_corners[key]) for key in bounding_boxes]
    if any(inside_roi):
        if inside_roi[0] and inside_roi[1]:
            return bounding_boxes[0]
        return bounding_boxes[inside_roi.index(True)]
    return 'out_of_roi'


def determine_block(start_time, end_time, startS, stopS):
    if start_time < startS[0] or end_time > stopS[-1]:
        return 'discard'
    for i, (run_start, run_stop) in enumerate(zip(startS, stopS), start=1):
        if start_time >= run_start and end_time <= run_stop:
            return 'mon_down'
        elif i < len(startS) and end_time <= startS[i]:
            return 'mon_up'
    return 'discard'





def save_fixation_and_saccade_results(processed_data_dir, fix_timepos_df, saccades, params):
    """
    Saves fixation and saccade results to files.
    Parameters:
    - processed_data_dir (str): Directory to save processed data.
    - all_fix_timepos (pd.DataFrame): DataFrame of fixation time positions.
    - fix_detection_results (list): List of fixation detection results.
    - saccade_detection_results (list): List of saccade detection results.
    - params (dict): Dictionary of parameters.
    """
    output_dir = processed_data_dir
    fixations_file = os.path.join(output_dir, f"all_fixations_and_saccades.pkl")
    logging.info("Saving: " + fixations_file)
    with open(fixations_file, 'wb') as f:
        pickle.dump((fix_timepos_df, saccades), f)







