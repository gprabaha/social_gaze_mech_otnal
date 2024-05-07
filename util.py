#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:50:13 2024

@author: prabaha
"""

import os
import numpy as np
from scipy.optimize import curve_fit
from math import degrees, atan2

import defaults

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


def calculate_roi_bounding_boxes(m1_landmarks, monitor_info=None):
    
    # for the face roi, just use the 4 extremeties of the far plane calibration 
    
    if monitor_info is None:
        monitor_info = defaults.fetch_monitor_info() if hasattr(defaults, 'fetch_monitor_info') else None
    if monitor_info is None:
        raise ValueError("Monitor info is required for conversion.")
    # Fetching coordinates in pixels
    left_eye = m1_landmarks['eyeOnLeft'][0][0][0]
    right_eye = m1_landmarks['eyeOnRight'][0][0][0]
    mouth = m1_landmarks['mouth'][0][0][0]
    lo_bottom_left = m1_landmarks[0]['leftObject'][0][0]['bottomLeft'][0][0]
    lo_top_right = m1_landmarks[0]['leftObject'][0][0]['topRight'][0][0]
    ro_bottom_left = m1_landmarks[0]['rightObject'][0][0]['bottomLeft'][0][0]
    ro_top_right = m1_landmarks[0]['rightObject'][0][0]['topRight'][0][0]
    # Calculate average y-coordinate of the eyes in pixels
    avg_eye_y_px = (left_eye[1] + right_eye[1]) / 2
    # Convert average eye y-coordinate to degrees of visual angle (DVA)
    avg_eye_y_deg = px2deg(avg_eye_y_px, monitor_info)
    # Calculate y-coordinates for eye bounding box
    eye_top_y_deg = avg_eye_y_deg - 1
    eye_bottom_y_deg = avg_eye_y_deg + 1
    # Calculate mean x-position of the eyes in pixels
    avg_eye_x_px = (left_eye[0] + right_eye[0]) / 2
    # Convert mean eye x-position to degrees of visual angle (DVA)
    avg_eye_x_deg = px2deg(avg_eye_x_px, monitor_info)
    # Calculate x-coordinates for eye bounding box
    eye_left_x_deg = avg_eye_x_deg - 2.5
    eye_right_x_deg = avg_eye_x_deg + 2.5
    # Calculate y-coordinates for face bounding box based on mouth position
    face_top_y_deg = avg_eye_y_deg + 4
    face_bottom_y_deg = avg_eye_y_deg + 1
    # Calculate x-coordinates for face bounding box
    face_left_x_deg = avg_eye_x_deg - 2.5
    face_right_x_deg = avg_eye_x_deg + 2.5
    # Create bounding boxes for left and right objects
    left_obj_bbox = {'bottomLeft': (lo_bottom_left[0], lo_bottom_left[1]), 'topRight': (lo_top_right[0], lo_top_right[1])}
    right_obj_bbox = {'bottomLeft': (ro_bottom_left[0], ro_bottom_left[1]), 'topRight': (ro_top_right[0], ro_top_right[1])}
    # Convert coordinates from degrees to pixels
    eye_bbox = {
        'bottomLeft': (deg2px(eye_left_x_deg, monitor_info), deg2px(eye_bottom_y_deg, monitor_info)),
        'topRight': (deg2px(eye_right_x_deg, monitor_info), deg2px(eye_top_y_deg, monitor_info))
    }
    face_bbox = {
        'bottomLeft': (deg2px(face_left_x_deg, monitor_info), deg2px(face_bottom_y_deg, monitor_info)),
        'topRight': (deg2px(face_right_x_deg, monitor_info), deg2px(face_top_y_deg, monitor_info))
    }
    return eye_bbox, face_bbox, left_obj_bbox, right_obj_bbox


def px2deg(px, monitor_info=None):
    if monitor_info is None:
        monitor_info = defaults.fetch_monitor_info() # in defaults
    h = monitor_info['height']
    d = monitor_info['distance']
    r = monitor_info['vertical_resolution']
    deg_per_px = degrees(atan2(0.5 * h, d)) / (0.5 * r)
    deg = px * deg_per_px
    return deg


def deg2px(deg, monitor_info=None):
    if monitor_info is None:
        monitor_info = defaults.fetch_monitor_info()  # in defaults
    h = monitor_info['height']
    d = monitor_info['distance']
    r = monitor_info['vertical_resolution']
    deg_per_px = degrees(atan2(0.5 * h, d)) / (0.5 * r)
    px = deg / deg_per_px
    return px


def create_timevec(n_samples, sampling_rate):
    return [i * sampling_rate for i in range(n_samples)]


def find_islands(binary_vec, min_samples=0):
    islands = []
    island_started = False
    island_start = 0
    for i, val in enumerate(binary_vec):
        if val == 1 and not island_started:
            island_started = True
            island_start = i
        elif val == 0 and island_started:
            island_started = False
            if i - island_start >= min_samples:
                islands.append([island_start, i - 1])
    # If the last island continues to the end of the array
    if island_started:
        if len(binary_vec) - island_start >= min_samples:
            islands.append([island_start, len(binary_vec) - 1])
    return np.array(islands)


def get_duration(start_stop):
    """
    Calculate duration from start and stop indices.
    Parameters:
    - start_stop (tuple): A tuple containing start and stop indices.
    Returns:
    - duration (int): Duration calculated from start and stop indices.
    """
    start, stop = start_stop
    duration = stop - start
    return duration


def identify_outliers(data, window_size=500, stride=250, threshold=None,
                      degree=10):
    num_points = data.shape[0]
    outlier_indices = []
    for i in range(0, num_points - window_size + 1, stride):
        window_data = data[i:i+window_size]
        curve_params = fit_curve(window_data[:, 0], window_data[:, 1], degree)
        if threshold is None:
            threshold = 10  # Default threshold in pixels
        window_outliers = _identify_outliers(window_data[:, 0], window_data[:, 1], curve_params, threshold)
        # Adjust outlier indices to global indices
        window_outliers += i
        # Remove outliers already identified in previous windows
        window_outliers = [idx for idx in window_outliers if idx not in outlier_indices]
        outlier_indices.extend(window_outliers)
    return outlier_indices

def _identify_outliers(x, y, curve_params, threshold):
    distances = calculate_distances(x, y, curve_params)
    outlier_indices = np.where(distances > threshold)[0]
    return outlier_indices

def fit_curve(x, y, degree):
    # Initial guess for coefficients (all zeros)
    initial_guess = [0.0] * (degree + 1)
    # Fit curve to the data
    coefficients, _ = curve_fit(polynomial_curve, x, y, p0=initial_guess)
    return coefficients

def polynomial_curve(x, *coefficients):
    return np.polyval(coefficients, x)

def calculate_distances(x, y, curve_params):
    curve_y = polynomial_curve(x, *curve_params)
    distances = np.abs(y - curve_y)
    return distances

def distance2p(x1, y1, x2, y2):
    """
    Calculate the distance between two points.
    Args:
    x1, y1: Coordinates of the first point.
    x2, y2: Coordinates of the second point.
    Returns:
    The distance between the two points.
    """
    dx = x2 - x1
    dy = y2 - y1
    distance2p = np.sqrt(dx**2 + dy**2)
    return distance2p

