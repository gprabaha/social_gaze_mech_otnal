#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:00:42 2024

@author: pg496
"""

import os
import glob
import scipy.io
import mat73
import numpy as np
import pandas as pd
import pickle

import util

import pdb


def get_monkey_and_dose_data(session_path):
    """
    Extracts information data from session path.
    Parameters:
    - session_path (str): Path to the session directory.
    Returns:
    - info_dict (dict): Dictionary containing information data.
    """
    file_list_info = glob.glob(f"{session_path}/*metaInfo.mat")
    if len(file_list_info) != 1:
        print(f"\nWarning: No metaInfo or more than one metaInfo found in folder: {session_path}.")
        return {'OT_dose': None, 'NAL_dose': None}
    try:
        data_info = scipy.io.loadmat(file_list_info[0])
        info = data_info.get('info', [None])[0]
        if info is not None:
            return {
                'monkey_1': info['monkey_1'][0][0],
                'monkey_2': info['monkey_2'][0][0],
                'OT_dose': float(info['OT_dose'][0][0]),
                'NAL_dose': float(info['NAL_dose'][0][0])
            }
    except Exception as e:
        print(f"\nError loading meta_info for folder: {session_path}: {e}")
    return {'OT_dose': None, 'NAL_dose': None}


def get_runs_data(session_path):
    """
    Extracts runs data from session path.
    Parameters:
    - session_path (str): Path to the session directory.
    Returns:
    - runs_dict (dict): Dictionary containing runs data.
    """
    file_list_runs = glob.glob(f"{session_path}/*runs.mat")
    if len(file_list_runs) != 1:
        print(f"\nWarning: No runs found in folder: {session_path}.")
        return {}
    try:
        data_runs = scipy.io.loadmat(file_list_runs[0])
        runs = data_runs.get('runs', [None])[0]
        if runs is not None:
            startS = [run['startS'][0][0] for run in runs]
            stopS = [run['stopS'][0][0] for run in runs]
            return {'startS': startS,
                    'stopS': stopS,
                    'num_runs': len(startS)}
    except Exception as e:
        print(f"\nError loading runs for folder: {session_path}: {e}")
    return {}


def load_farplane_cal_and_get_bl_and_tr_roi_coords_m1(session_path, params):
    """
    Extracts M1 ROI bounding boxes from session path.
    Parameters:
    - session_path (str): Path to the session directory.
    - params (dict): Dictionary containing parameters including session path and map_roi_coord_to_eyelink_space flag.
    Returns:
    - bbox_dict (dict): Dictionary containing M1 ROI bounding boxes.
    """
    file_list_m1_landmarks = glob.glob(f"{session_path}/*M1_farPlaneCal.mat")
    if len(file_list_m1_landmarks) != 1:
        print(f"\nWarning: No m1_landmarks or more than one landmarks found in folder: {session_path}.")
        return {'eye_bbox': None,
                'face_bbox': None,
                'left_obj_bbox': None,
                'right_obj_bbox': None}
    try:
        data_m1_landmarks = scipy.io.loadmat(file_list_m1_landmarks[0])
        m1_landmarks = data_m1_landmarks.get('farPlaneCal', None)
        if m1_landmarks is not None:
            return util.get_bl_and_tr_roi_coords_m1(m1_landmarks, params)
    except Exception as e:
        print(f"\nError loading m1_landmarks for folder: {session_path}: {e}")
    return {'eye_bbox': None,
            'face_bbox': None,
            'left_obj_bbox': None,
            'right_obj_bbox': None}


def get_labelled_gaze_positions_dict_m1(idx, params):
    """
    Process gaze data from a session folder.
    Parameters:
    - idx (int): Index to access specific session data.
    - params (dict): Dictionary containing session information.
    Returns:
    - gaze_data (tuple): Tuple containing gaze positions and associated metadata.
    """
    session_paths = params['session_paths']
    meta_info_list = params['meta_info_list']
    session_categories = params['session_categories']
    map_gaze_pos_coord_to_eyelink_space = params.get('map_gaze_pos_coord_to_eyelink_space', False)
    folder_path = session_paths[idx]
    mat_files = [f for f in os.listdir(folder_path) if 'M1_gaze.mat' in f]
    if len(mat_files) != 1:
        print(f"\nError: Multiple or no '*_M1_gaze.mat' files found in folder: {folder_path}")
        return None
    mat_file_path = os.path.join(folder_path, mat_files[0])
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        sampling_rate = float(mat_data['M1FS'])
        M1Xpx = mat_data['M1Xpx'].squeeze()
        M1Ypx = mat_data['M1Ypx'].squeeze()
        coordinates = np.column_stack((M1Xpx, M1Ypx))
        coordinates_inverted_y = util.remap_source_coords(
            coordinates, params, 'inverted_to_standard_y_axis')
        gaze_positions = util.remap_source_coords(
            coordinates_inverted_y, params, 'to_eyelink_space')
        meta_info = meta_info_list[idx]
        meta_info.update({'sampling_rate': sampling_rate,
                          'category': session_categories[idx]})
        return gaze_positions, meta_info
    except Exception as e:
        print(f"\nError loading file '{mat_files[0]}': {e}")
        return None


def load_labelled_gaze_positions(params):
    """
    Load labelled gaze positions from pickle file.
    Parameters:
    - params (dict): Dictionary containing root data directory and other parameters.
    Returns:
    - labelled_gaze_positions (tuple): Tuple containing gaze positions and associated metadata.
    """
    processed_data_dir = params['processed_data_dir']
    # Adjusted file name based on flags
    flag_info = util.get_filename_flag_info(params)
    file_name = f'labelled_gaze_positions_m1{flag_info}.pkl'
    with open(os.path.join(processed_data_dir, file_name), 'rb') as f:
        return pickle.load(f)
    

def load_m1_fixations(params):
    """
    Load M1 fixations and related data.
    Parameters:
    - params (dict): Dictionary containing root data directory and other parameters.
    Returns:
    - fixations_df (pd.DataFrame): DataFrame containing M1 fixations and their time positions.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    # Load fixations time positions
    fix_timepos_file_name = f'fix_timepos_m1{flag_info}.csv'
    fix_timepos_m1 = pd.read_csv(os.path.join(
        processed_data_dir, fix_timepos_file_name))
    return fix_timepos_m1


def load_fix_detection_results(params):
    """
    Loads fixation detection results from file.
    Parameters:
    - params (dict): Dictionary of parameters.
    Returns:
    - fix_detection_results (list): List of fixation detection results.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    results_file_name = f'fixation_session_results_m1{flag_info}.npz'
    file_path = os.path.join(processed_data_dir, results_file_name)
    if os.path.exists(file_path):
        # Load the .npz file
        with np.load(file_path, allow_pickle=True) as data:
            fixations_list = data['fixations']
            info_list = data['info']
        # Load fixations time positions from CSV
        fix_timepos_file_name = f'fix_timepos_m1{flag_info}.csv'
        timepos_df = pd.read_csv(os.path.join(
            processed_data_dir, fix_timepos_file_name))
        timepos_list = [timepos_df] * len(fixations_list)  # Assuming timepos_list is a list of identical DataFrames
        # Reconstruct the fix_detection_results list
        fix_detection_results = [(fixations_list[i], timepos_list[i], info_list[i]) 
                                 for i in range(len(fixations_list))]
        return fix_detection_results
    else:
        print(f"File {file_path} does not exist.")
        return None


def load_m1_fixation_labels(params):
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    # Load fixation labels
    fixation_labels_file_name = f'fixation_labels_m1{flag_info}.csv'
    fixation_labels_m1 = pd.read_csv(os.path.join(
        processed_data_dir, fixation_labels_file_name))
    return fixation_labels_m1


def load_saccade_labels(params):
    """
    Loads the labelled saccades from a specified directory.
    Parameters:
    - params (dict): Dictionary containing parameters including the load directory.
    Returns:
    - labelled_saccades (DataFrame): DataFrame containing saccade information with labels.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    file_path = os.path.join(processed_data_dir, f'labelled_saccades_{flag_info}.csv')
    
    if os.path.exists(file_path):
        labelled_saccades = pd.read_csv(file_path)
        print(f"Saccade labels loaded from {file_path}")
        return labelled_saccades
    else:
        print(f"No file found at {file_path}")
        return None



def get_spiketimes_and_labels_for_one_session(session_path):
    """
    Extracts spike times and labels from a session.
    Parameters:
    - session_path (str): Path to the session.
    Returns:
    - session_spikeTs_s (list): List of spike times in seconds.
    - session_spikeTs_ms (list): List of spike times in milliseconds.
    - spike_df (DataFrame): DataFrame containing spike labels.
    """
    session_spikeTs_s = []
    session_spikeTs_ms = []
    session_spikeTs_labels = []
    label_cols = ['session_name', 'channel', 'channel_label',
                  'unit_no_within_channel', 'unit_label', 'uuid',
                  'n_spikes', 'region']
    session_name = os.path.basename(os.path.normpath(session_path))
    file_list_spikeTs = glob.glob(f"{session_path}/*spikeTs_regForm.mat")
    if len(file_list_spikeTs) != 1:
        print(f"\nWarning: No spikeTs or more than one spikeTs found in folder: {session_path}.")
        return session_spikeTs_s, session_spikeTs_ms, pd.DataFrame(columns=label_cols)
    try:
        data_spikeTs = mat73.loadmat(file_list_spikeTs[0])
        spikeTs_struct = data_spikeTs['spikeTs'][0]
        for unit in spikeTs_struct:
            spikeTs_sec = np.squeeze(unit['spikeS'])
            spikeTs_ms = np.squeeze(unit['spikeMs'])
            session_spikeTs_s.append(spikeTs_sec.tolist())  # Append spike times in seconds
            session_spikeTs_ms.append(spikeTs_ms.tolist())  # Append spike times in milliseconds
            chan = unit['chan'][0][0]
            chan_label = unit['chanStr'][0]
            unit_no_in_channel = unit['unit'][0][0]
            unit_label = unit['unitStr'][0]
            uuid = unit['UUID'][0]
            n_spikes = unit['spikeN'][0][0]
            region = unit['region'][0]
            # Append label row to the labels list
            session_spikeTs_labels.append(
                [session_name, chan, chan_label, unit_no_in_channel,
                 unit_label, uuid, n_spikes, region])
        # Create DataFrame
        spike_df = pd.DataFrame(session_spikeTs_labels, columns=label_cols)
        return session_spikeTs_s, session_spikeTs_ms, spike_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], pd.DataFrame(columns=label_cols)


def load_processed_spike_data(params):
    """
    Load processed spike data from files.
    Parameters:
    - params (dict): Dictionary containing root data directory and other parameters.
    Returns:
    - spikeTs_s (list): List of spike times in seconds.
    - spikeTs_ms (list): List of spike times in milliseconds.
    - spike_labels (DataFrame): DataFrame containing spike labels.
    """
    root_data_dir = params.get('root_data_dir')
    flag_info = util.get_filename_flag_info(params)
    # Load spikeTs_s
    spiketimes_s_path = os.path.join(root_data_dir, f'spiketimes_s{flag_info}.pkl')
    with open(spiketimes_s_path, 'rb') as f:
        spikeTs_s = pickle.load(f)
    # Load spikeTs_ms
    spiketimes_ms_path = os.path.join(root_data_dir, f'spiketimes_ms{flag_info}.pkl')
    with open(spiketimes_ms_path, 'rb') as f:
        spikeTs_ms = pickle.load(f)
    # Load spike labels
    labels_path = os.path.join(root_data_dir, f'spike_labels{flag_info}.csv')
    spike_labels = pd.read_csv(labels_path)
    return spikeTs_s, spikeTs_ms, spike_labels

