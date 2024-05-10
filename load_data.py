#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:00:42 2024

@author: pg496
"""

import os
import glob
import scipy
import numpy as np

import util

import pdb


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


### Function to get information data
def get_monkey_and_dose_data(session_path):
    """
    Extracts information data from session path.
    Parameters:
    - session_path (str): Path to the session.
    Returns:
    - info_dict (dict): Dictionary containing information data.
    """
    file_list_info = glob.glob(f"{session_path}/*metaInfo.mat")
    if len(file_list_info) != 1:
        print(f"Warning: No metaInfo or more than one metaInfo found in folder: {session_path}.")
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
    except Exception as e:
        print(f"Error loading meta_info for folder: {session_path}: {e}")



### Function to get runs data
def get_runs_data(session_path):
    """
    Extracts runs data from session path.
    Parameters:
    - session_path (str): Path to the session.
    Returns:
    - runs_dict (dict): Dictionary containing runs data.
    """
    file_list_runs = glob.glob(f"{session_path}/*runs.mat")
    if len(file_list_runs) != 1:
        print(f"Warning: No runs found in folder: {session_path}.")
    try:
        data_runs = scipy.io.loadmat(file_list_runs[0])
        runs = data_runs.get('runs', None)
        if runs is not None:
            startS = [run['startS'][0][0] for run in runs[0]]
            stopS = [run['stopS'][0][0] for run in runs[0]]
            num_runs = len(startS)
            return {'startS': startS, 'stopS': stopS, 'num_runs': num_runs}
    except Exception as e:
        print(f"Error loading runs for folder: {session_path}: {e}")


def get_labelled_gaze_positions_dict_m1(folder_path, meta_info_list, session_categories, idx):
    """
    Process gaze data from a session folder.
    Parameters:
    - folder_path (str): Path to the session folder.
    - meta_info_list (list): List of dictionaries containing meta-information for each session.
    - session_categories (ndarray): Session categories.
    Returns:
    - gaze_data (tuple): Tuple containing gaze positions and associated metadata.
    """
    mat_files = [f for f in os.listdir(folder_path) if 'M1_gaze.mat' in f]
    if len(mat_files) != 1:
        print(f"Error: Multiple or no '*_M1_gaze.mat' files found in folder: {folder_path}")
        return None
    mat_file = mat_files[0]
    mat_file_path = os.path.join(folder_path, mat_file)
    mat_file_name = os.path.basename(mat_file_path)
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        sampling_rate = float(mat_data['M1FS'])
        M1Xpx = mat_data['M1Xpx'].squeeze()
        M1Ypx = mat_data['M1Ypx'].squeeze()
        gaze_positions = np.array(np.column_stack((M1Xpx, M1Ypx)))
        meta_info = meta_info_list[idx]
        meta_info.update({'sampling_rate': sampling_rate, 'category': session_categories[idx]})
        return gaze_positions, meta_info
    except Exception as e:
        print(f"Error loading file '{mat_file_name}': {str(e)}")
        return None


### Function to get M1 ROI bounding boxes
def get_m1_roi_bounding_boxes(session_path):
    """
    Extracts M1 ROI bounding boxes from session path.
    Parameters:
    - session_path (str): Path to the session.
    Returns:
    - bbox_dict (dict): Dictionary containing M1 ROI bounding boxes.
    """
    file_list_m1_landmarks = glob.glob(f"{session_path}/*M1_farPlaneCal.mat")
    if len(file_list_m1_landmarks) != 1:
        print(f"Warning: No m1_landmarks or more than one landmarks found in folder: {session_path}.")
        return {'eye_bbox': None, 'face_bbox': None, 'left_obj_bbox': None, 'right_obj_bbox': None}
    try:
        data_m1_landmarks = scipy.io.loadmat(file_list_m1_landmarks[0])
        m1_landmarks = data_m1_landmarks.get('farPlaneCal', None)
        if m1_landmarks is not None:
            eye_bbox, face_bbox, left_obj_bbox, right_obj_bbox = util.calculate_roi_bounding_box_corners(m1_landmarks)
            return {'eye_bbox': eye_bbox, 'face_bbox': face_bbox, 'left_obj_bbox': left_obj_bbox, 'right_obj_bbox': right_obj_bbox}
        else:
            return {'eye_bbox': None, 'face_bbox': None, 'left_obj_bbox': None, 'right_obj_bbox': None}
    except Exception as e:
        print(f"Error loading m1_landmarks for folder: {session_path}: {e}")
        return {'eye_bbox': None, 'face_bbox': None, 'left_obj_bbox': None, 'right_obj_bbox': None}


#### Workng on this
def get_spiketimes_and_labels_for_one_session(session_path):
    session_spikeTs_s = []
    session_spikeTs_ms = []
    session_spikeTs_labels = []
    label_cols = []
    session_name =  os.path.basename(os.path.normpath(session_path))
    file_list_spikeTs = glob.glob(f"{session_path}/*spikeTs.mat")
    if len(file_list_spikeTs) != 1:
        print(f"Warning: No runs found in folder: {session_path}.")
    try:
        data_spikeTs = scipy.io.loadmat(file_list_spikeTs[0])
        pdb.set_trace()
        spikeTs_struct = data_spikeTs.get('spikeTs_struct', None)
        # Edit here to get all the spiketimes and labels
        return session_spikeTs_s, session_spikeTs_ms, session_spikeTs_labels
    except Exception as e:
        return session_spikeTs_s, session_spikeTs_ms, session_spikeTs_labels


def get_runs_data_copy(session_path):
    """
    Extracts runs data from session path.
    Parameters:
    - session_path (str): Path to the session.
    Returns:
    - runs_dict (dict): Dictionary containing runs data.
    """
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
