#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:36:36 2024

@author: prabaha
"""

import numpy as np
from util import *

############################################################################################
def extract_labelled_gaze_positions(unique_doses, dose_inds, meta_info_list, session_paths, session_categories):
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
        for idx in indices_list:
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
                mat_data = loadmat(mat_file_path)
                sampling_rate = float(mat_data['M1FS'])
                M1Xpx = mat_data['M1Xpx'].squeeze()  # Squeeze to remove singleton dimensions
                M1Ypx = mat_data['M1Ypx'].squeeze()
                
                # Convert x and y positions to a single array
                gaze_positions = np.array(np.column_stack((M1Xpx, M1Ypx)))
                
                # Append gaze positions and associated metadata to the list
                meta_info = meta_info_list[idx]  # Copy to avoid modifying the original meta_info
                meta_info.update({'sampling_rate': sampling_rate, 'category': session_categories[idx]})  # Add sampling rate to metadata
                labelled_gaze_positions.append((gaze_positions, meta_info))
                print(f"Loaded file: {mat_file_name}")
            except Exception as e:
                print(f"Error loading file '{mat_file_name}': {str(e)}")
    
    return labelled_gaze_positions


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
    #print(f"Min vel: {min(vel_norm)}, max_Vel: {max(vel_norm)}, thresholds: {vel_thresh}")
    
    starts_stops = find_islands(above_thresh, min_samples)
    print(start_stops)

    return start_stops

def extract_saccade_positions(run_positions, saccade_start_stops):
    saccades = []
    for start, stop in saccade_start_stops:
        saccade = run_positions[start:stop+1, :]
        saccades.extend(saccade)
    return saccades


def extract_saccades_with_labels(labelled_gaze_positions):
    saccade_params = fetch_default_saccade_pars()
    vel_thresh = saccade_params['vel_thresh']
    min_samples = saccade_params['min_samples']
    smooth_func = saccade_params['smooth_func']
    saccades = []
    saccade_labels = []
    for i, session in enumerate(labelled_gaze_positions):
        print(f"Extracting saccades for session: {i+1}/{len(labelled_gaze_positions)}")
        positions = session[0]
        info = session[1]
        sampling_rate = info['sampling_rate']
        n_samples = positions.shape[0]
        time_vec = create_timevec(n_samples, sampling_rate)
        category = info['category']
        n_runs = info['num_runs']
        for run in range(n_runs):
            run_start = info['startS'][run]
            run_stop = info['stopS'][run]
            run_time = (time_vec > run_start) & (time_vec <= run_stop)
            run_positions = positions[run_time,:]
            run_x = px2deg(run_positions[:,0].T)
            run_y = px2deg(run_positions[:,1].T)
            saccade_start_stops = find_saccades(run_x, run_y, sampling_rate, vel_thresh, min_samples, smooth_func)
            saccades_in_run = extract_saccade_positions(run_positions, saccade_start_stops)
            n_saccades = len(saccades_in_run)
            print(f"n_Saccades: {n_saccades}")
            saccades.extend(saccades_in_run)
            saccade_labels.extend([[category, i, run]] * n_saccades)
    assert len(saccades) == len(saccade_labels)
    return saccades, saccade_labels