#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:40:06 2024

@author: pg496
"""

import numpy as np
import os

import util


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