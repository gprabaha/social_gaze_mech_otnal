#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:36:36 2024

@author: prabaha
"""

import numpy as np
from tqdm import tqdm
import os
import scipy.io
import glob
from multiprocessing import Pool

import util
import defaults
import fix

import pdb

###
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
        meta_info = {'session_name': os.path.basename(os.path.normpath(session_path))}
        meta_info.update(get_info_data(session_path))
        meta_info.update(get_runs_data(session_path))
        meta_info.update(get_m1_landmarks_data(session_path))
        meta_info_list.append(meta_info)
    return meta_info_list

def get_info_data(session_path):
    file_list_info = glob.glob(f"{session_path}/*metaInfo.mat")
    if len(file_list_info) != 1:
        print(f"Warning: No metaInfo found in folder: {session_path}.")
        return {'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None}
    try:
        data_info = scipy.io.loadmat(file_list_info[0])
        info = data_info.get('info', None)
        if info is not None:
            info = info[0][0]
            return {
                'monkey_1': info['monkey_1'][0],
                'monkey_2': info['monkey_2'][0],
                'OT_dose': float(info['OT_dose'][0]),
                'NAL_dose': float(info['NAL_dose'][0])
            }
        else:
            return {'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None}
    except Exception as e:
        print(f"Error loading meta_info for folder: {session_path}: {e}")
        return {'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None}


def get_runs_data(session_path):
    file_list_runs = glob.glob(f"{session_path}/*runs.mat")
    if len(file_list_runs) != 1:
        print(f"Warning: No runs found in folder: {session_path}.")
        return {'startS': None, 'stopS': None, 'num_runs': 0}
    try:
        data_runs = scipy.io.loadmat(file_list_runs[0])
        runs = data_runs.get('runs', None)
        if runs is not None:
            startS = [run['startS'][0][0] for run in runs[0]]
            stopS = [run['stopS'][0][0] for run in runs[0]]
            num_runs = len(startS)
            return {'startS': startS, 'stopS': stopS, 'num_runs': num_runs}
        else:
            return {'startS': None, 'stopS': None, 'num_runs': 0}
    except Exception as e:
        print(f"Error loading runs for folder: {session_path}: {e}")
        return {'startS': None, 'stopS': None, 'num_runs': 0}


def get_m1_landmarks_data(session_path):
    file_list_m1_landmarks = glob.glob(f"{session_path}/*M1_farPlaneCal.mat")
    if len(file_list_m1_landmarks) != 1:
        print(f"Warning: No m1_landmarks found in folder: {session_path}.")
        return {'eye_bbox': None, 'face_bbox': None, 'left_obj_bbox': None, 'right_obj_bbox': None}
    try:
        data_m1_landmarks = scipy.io.loadmat(file_list_m1_landmarks[0])
        m1_landmarks = data_m1_landmarks.get('farPlaneCal', None)
        if m1_landmarks is not None:
            eye_bbox, face_bbox, left_obj_bbox, right_obj_bbox = util.calculate_roi_bounding_boxes(m1_landmarks)
            return {'eye_bbox': eye_bbox, 'face_bbox': face_bbox, 'left_obj_bbox': left_obj_bbox, 'right_obj_bbox': right_obj_bbox}
        else:
            return {'eye_bbox': None, 'face_bbox': None, 'left_obj_bbox': None, 'right_obj_bbox': None}
    except Exception as e:
        print(f"Error loading m1_landmarks for folder: {session_path}: {e}")
        return {'eye_bbox': None, 'face_bbox': None, 'left_obj_bbox': None, 'right_obj_bbox': None}


###
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


###
def extract_labelled_gaze_positions_m1(unique_doses, dose_inds, meta_info_list, session_paths, session_categories):
    """
    Extracts labelled gaze positions from files associated with unique doses.
    Parameters:
    - unique_doses (ndarray): Unique dose combinations.
    - dose_inds (list): List of lists containing indices for each unique dose.
    - meta_info_list (list): List of dictionaries containing meta-information for each session.
    - session_paths (list): List of paths to sessions.
    Returns:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    """
    labelled_gaze_positions = []
    # Iterate over unique doses and associated indices
    for dose, indices_list in zip(unique_doses, dose_inds):
        for idx in tqdm(indices_list, desc="Processing indices for dose", unit="index"):
            folder_path = session_paths[idx]
            # Assuming there's only one file with extension M1_gaze.mat in each folder
            mat_files = [f for f in os.listdir(folder_path) if 'M1_gaze.mat' in f]
            if len(mat_files) != 1:
                print(f"Error: Multiple or no '*_M1_gaze.mat' files found in folder: {folder_path}")
                continue
            mat_file = mat_files[0]
            mat_file_path = os.path.join(folder_path, mat_file)
            mat_file_name = os.path.basename(mat_file_path)
            # Load *_M1_gaze.mat file
            try:
                mat_data = scipy.io.loadmat(mat_file_path)
                sampling_rate = float(mat_data['M1FS'])
                M1Xpx = mat_data['M1Xpx'].squeeze()  # Squeeze to remove singleton dimensions
                M1Ypx = mat_data['M1Ypx'].squeeze()
                # Convert x and y positions to a single array
                gaze_positions = np.array(np.column_stack((M1Xpx, M1Ypx)))
                # Append gaze positions and associated metadata to the list
                meta_info = meta_info_list[idx]  # Copy to avoid modifying the original meta_info
                meta_info.update({'sampling_rate': sampling_rate, 'category': session_categories[idx]})  # Add sampling rate to metadata
                labelled_gaze_positions.append((gaze_positions, meta_info))
            except Exception as e:
                print(f"Error loading file '{mat_file_name}': {str(e)}")
    return labelled_gaze_positions


###
def extract_saccades_with_labels(labelled_gaze_positions):
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
            #import pdb; pdb.set_trace()
            saccade_labels.extend([[category, session_identifier, run]] * n_saccades)
    assert len(saccades) == len(saccade_labels)
    return saccades, saccade_labels

def find_saccades(x, y, sr, vel_thresh, min_samples, smooth_func):
    """
    Find start and stop indices of saccades.
    Parameters:
    - x: array-like, x-coordinates of eye movements
    - y: array-like, y-coordinates of eye movements
    - sr: float, sampling rate
    - vel_thresh: float, minimum velocity threshold for saccade onset
    - min_samples: int, minimum duration of a saccade in samples
    - smooth_func: function, function for smoothing input data
    Returns:
    - start_stops: list of arrays, start and stop indices of saccades for each trial
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

def extract_saccade_positions(run_positions, saccade_start_stops):
    saccades = []
    for start, stop in saccade_start_stops:
        saccade = run_positions[start:stop+1, :]
        saccades.append(saccade)
    return saccades



def extract_fixations_with_labels_parallel(labelled_gaze_positions, parallel=True):
    session_identifiers = list(range(len(labelled_gaze_positions)))
    sessions = [(i, session[0], session[1]) for i, session in enumerate(labelled_gaze_positions)]
    if parallel:
        with Pool() as pool:
            results = list(tqdm(pool.imap(get_session_fixations, sessions),
                                total=len(sessions), desc="Extracting fixations in parallel", unit="session"))
    else:
        results = [get_session_fixations(session) for session in
                   tqdm(sessions, desc="Extracting fixations in serial", unit="session")]
    all_fixations = []
    all_fixation_labels = []
    for session_fixations, session_labels in results:
        all_fixations.extend(session_fixations)
        all_fixation_labels.extend(session_labels)
    return all_fixations, all_fixation_labels


def get_session_fixations(session):
    session_identifier, positions, info = session
    sampling_rate = info['sampling_rate']
    n_samples = positions.shape[0]
    time_vec = util.create_timevec(n_samples, sampling_rate)
    category = info['category']
    n_runs = info['num_runs']
    n_intervals = n_runs - 1
    print(f"\nDetecting all fixations in: {info['session_name']}")
    fix_vec_entire_session = fix.is_fixation(util.px2deg(positions), time_vec, sampling_rate=sampling_rate)
    fixations = util.find_islands(fix_vec_entire_session)
    fixation_labels = []
    print(f"\nLabelling fixations for: {info['session_name']}")
    for start_stop in fixations:
        duration = util.get_duration(start_stop)
        run, block , fix_roi = detect_run_block_and_roi(start_stop, positions, info)
        agent = info['monkey_1']
        # Construct the details for the current fixation
        fixation_info = [category, session_identifier, run, block, duration, fix_roi, agent]
        fixation_labels.append(fixation_info)
    assert fixations.shape[0] == len(fixation_labels)
    return fixations, fixation_labels


def detect_run_block_and_roi(start_stop, positions, info):
    """
    Detect the run, block, and ROI based on start and stop indices, gaze positions, and session info.
    Parameters:
    - start_stop (tuple): A tuple containing start and stop indices.
    - positions (numpy.ndarray): Array containing gaze positions.
    - info (dict): Dictionary containing session information.
    Returns:
    - run (int or None): Detected run number or None.
    - block (str): Detected block identifier.
    - fix_roi (str): Detected region of interest.
    """
    start, stop = start_stop
    
    # Detect run and block
    startS = info.get('startS', [])
    stopS = info.get('stopS', [])
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
    # Find fixation ROI
    fix_pos = positions[start:stop, :]
    mean_fix_pos = np.nanmean(fix_pos, axis=0)
    bounding_boxes = ['eye_bbox', 'face_bbox', 'left_obj_bbox', 'right_obj_bbox']
    for key in bounding_boxes:
        value = info.get(key)
        if value:
            bottom_left = value['bottomLeft']
            top_right = value['topRight']
            if bottom_left[0] <= mean_fix_pos[0] <= top_right[0] and \
               bottom_left[1] <= mean_fix_pos[1] <= top_right[1]:
                fix_roi = key
                break
    else:
        fix_roi = 'out_of_roi' 
    return run, block, fix_roi
