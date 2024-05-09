#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:50:13 2024

@author: prabaha
"""

import os
import numpy as np
from scipy.optimize import curve_fit
from math import degrees, atan2, sqrt

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



def calculate_roi_bounding_box_corners(m1_landmarks):
    # Positive Y axis is downward so that needs to be accounted for
    # Face bounding box is just the one described the four corner points
    corner_name_order = ['topLeft', 'topRight', 'bottomRight', 'bottomLeft']
    left_eye = m1_landmarks['eyeOnLeft'][0][0][0]
    right_eye = m1_landmarks['eyeOnRight'][0][0][0]
    eye_bb_corners = construct_eye_bounding_box(left_eye, right_eye, corner_name_order)
    face_bb_corners = {key: m1_landmarks[key][0][0][0] for key in corner_name_order}
    left_obj_bb_corners = {key: m1_landmarks['leftObject'][0][0][0][key][0][0] for key in corner_name_order}
    right_obj_bb_corners = {key: m1_landmarks['rightObject'][0][0][0][key][0][0] for key in corner_name_order}
    return eye_bb_corners, face_bb_corners, left_obj_bb_corners, right_obj_bb_corners


def construct_eye_bounding_box(left_eye, right_eye, corner_name_order):
    inter_eye_dist = distance(left_eye, right_eye)
    offset = inter_eye_dist / 2
    # Coordinates of the right eye
    right_eye_x, right_eye_y = right_eye
    # Coordinates of the left eye
    left_eye_x, left_eye_y = left_eye
    corner_dict = {}
    for corner in corner_name_order:
        if corner == 'topLeft':
            corner_dict[corner] = (left_eye_x - offset, left_eye_y - offset)
        elif corner == 'topRight':
            corner_dict[corner] = (right_eye_x + offset, right_eye_y - offset)
        elif corner == 'bottomRight':
            corner_dict[corner] = (right_eye_x + offset, right_eye_y + offset)
        elif corner == 'bottomLeft':
            corner_dict[corner] = (left_eye_x - offset, left_eye_y + offset)
    return corner_dict


def is_inside_quadrilateral(point, corners, tolerance=1):
    # It is okay to have 1 square pixel error in area matching
    # This will avoid errord due to precision-related calculation mistakes
    # Very few points pretty much in the boundary might get included as a consequence
    x, y = point
    x1, y1 = corners['topLeft']
    x2, y2 = corners['topRight']
    x3, y3 = corners['bottomRight']
    x4, y4 = corners['bottomLeft']
    total_area = shoelace_formula(x1, y1, x2, y2, x3, y3) + \
                 shoelace_formula(x1, y1, x3, y3, x4, y4)
    area_point1 = shoelace_formula(x, y, x1, y1, x2, y2)
    area_point2 = shoelace_formula(x, y, x2, y2, x3, y3)
    area_point3 = shoelace_formula(x, y, x3, y3, x4, y4)
    area_point4 = shoelace_formula(x, y, x4, y4, x1, y1)
    return abs(total_area - (area_point1 + area_point2 + area_point3 + area_point4)) < tolerance


def shoelace_formula(x1, y1, x2, y2, x3, y3):
    x1 = np.float64(x1)
    y1 = np.float64(y1)
    x2 = np.float64(x2)
    y2 = np.float64(y2)
    x3 = np.float64(x3)
    y3 = np.float64(y3)
    return 0.5 * abs((x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1))


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_fix_positions(start_stop, positions):
    start, stop = start_stop
    return positions[start:stop,:]

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

