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

##################################################################
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
    num_trials = x.shape[0]
    
    start_stops = []

    for i in range(num_trials):
        x0 = smooth_func(x[i, :])
        y0 = smooth_func(y[i, :])
        
        vx = np.gradient(x0) * sr
        vy = np.gradient(y0) * sr
        
        vel_norm = np.sqrt(vx**2 + vy**2)  # Norm of velocity vector
        
        above_vel_thresh = vel_norm >= vel_thresh
        
        starts, durs = find_islands(above_vel_thresh)
        stops = starts + durs
        within_sample_thresh = durs >= min_samples
        starts = starts[within_sample_thresh]
        stops = stops[within_sample_thresh]
        starts, stops = merge_intervals(starts, stops)
        peak_velocities = peak_velocity(vel_norm, starts, stops)
        
        start_stops.append(np.column_stack((starts, stops, peak_velocities)))

    return start_stops


######################
def find_islands(vec):
    """
    Find starting indices of all contiguous groups of non-zero elements.
    
    Parameters:
    - vec: 1D array-like, input sequence
    
    Returns:
    - starts: list of starting indices of contiguous groups of non-zero elements
    - durs: list of durations (number of contiguous elements) for each group
    
    Example:
    starts, durs = find_islands([True, True, False, True])  # Returns [0, 3], [2, 1]
    """
    tsig = vec
    dsig = [0] + [1 if tsig[i+1] > tsig[i] else -1 for i in range(len(tsig) - 1)] + [0]
    starts = [i for i in range(len(dsig)) if dsig[i] > 0]
    ends = [i - 1 for i in range(len(dsig)) if dsig[i] < 0]
    durs = [end - start + 1 for start, end in zip(starts, ends)]
    return starts, durs

###########################################
def peak_velocity(vel_norm, starts, stops):
    """
    Calculate peak velocity for each saccade.

    Parameters:
    - vel_norm: array-like, norm of velocity vector
    - starts: array-like, start indices of saccades
    - stops: array-like, stop indices of saccades

    Returns:
    - peak_vel: array-like, peak velocity for each saccade
    """
    peak_vel = np.zeros(len(starts))

    for i, (start, stop) in enumerate(zip(starts, stops)):
        peak_vel[i] = np.max(vel_norm[start:stop+1])

    return peak_vel

###################################
def merge_intervals(starts, stops):
    """
    Merge overlapping intervals.

    Parameters:
    - starts: array-like, start indices of intervals
    - stops: array-like, stop indices of intervals

    Returns:
    - starts: array-like, merged start indices
    - stops: array-like, merged stop indices
    """
    if not starts:
        return starts, stops

    merged_starts = [starts[0]]
    merged_stops = [stops[0]]

    for i in range(1, len(starts)):
        if starts[i] <= merged_stops[-1]:
            merged_stops[-1] = stops[i]
        else:
            merged_starts.append(starts[i])
            merged_stops.append(stops[i])

    return np.array(merged_starts), np.array(merged_stops)


def extract_saccades_with_labels(labelled_gaze_positions):
    
    saccades = []
    saccade_labels = []
    for i, session in enumerate(labelled_gaze_positions):
        positions = session[0]
        info = session[1]
        category = info['category']
        if category == 0:
            continue
        else:
            n_runs = info['num_runs']
            
            
    return saccades, saccade_labels