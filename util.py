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
from datetime import datetime
import itertools

import defaults

import pdb


def fetch_root_data_dir(params):
    """
    Returns the root data directory based on whether it's running on a cluster or not.
    Parameters:
    - params (dict): Dictionary containing parameters.
    Returns:
    - root_data_dir (str): Root data directory path.
    """
    is_cluster = params['is_cluster']
    root_data_dir = "/gpfs/milgram/project/chang/pg496/data_dir/otnal/" if is_cluster \
        else "/Volumes/Stash/changlab/sorted_neural_data/social_gaze_otnal/AllFVProcessed/"
    params.update({'root_data_dir': root_data_dir})
    return root_data_dir, params


def fetch_data_source_dir(params):
    root_data_dir = params.get('root_data_dir')
    data_source_dir = os.path.join(root_data_dir, 'data_source_7point3')
    params.update({'data_source_dir': data_source_dir})
    return data_source_dir, params


def fetch_processed_data_dir(params):
    root_data_dir = params.get('root_data_dir')
    processed_data_dir = os.path.join(root_data_dir, 'processed_data')
    params.update({'processed_data_dir': processed_data_dir})
    return processed_data_dir, params


def fetch_session_subfolder_paths_from_source(params):
    """
    Retrieves subfolders within a given directory.
    Parameters:
    - params (dict): Dictionary containing parameters.
    Returns:
    - subfolders (list): List of subfolder paths.
    """
    data_source_dir = params['data_source_dir']
    session_paths = [f.path for f in os.scandir(data_source_dir)
                     if f.is_dir()]
    params.update({'session_paths': session_paths})
    return session_paths, params


def add_datestr_to_dir_path(path):
    # Get the current date as a string in YYYYMMDD format
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(path, date_str)


def remap_source_coords(coord, params, remapping_type):

    def remap_inverted_to_standard_y_axis(coord):
        def remap_single_coord_from_inverted_to_standard_y_axis(coord):
            return (coord[0], -coord[1])
        if not params.get(
                'remap_source_coord_from_inverted_to_standard_y_axis', False):
            return coord
        if isinstance(coord, (list, tuple)):
            if len(coord) == 2 and all(isinstance(i, (int, float))
                                       for i in coord):
                remapped_coord = \
                    remap_single_coord_from_inverted_to_standard_y_axis(coord)
                return type(coord)(remapped_coord)
            elif all(isinstance(i, (list, tuple, np.ndarray)) \
                     and len(i) == 2 for i in coord):
                return [remap_single_coord_from_inverted_to_standard_y_axis(c)
                        for c in coord]
        elif isinstance(coord, np.ndarray):
            if coord.ndim == 1 and coord.shape[0] == 2:
                return np.array(
                    remap_single_coord_from_inverted_to_standard_y_axis(coord))
            elif coord.ndim == 2 and coord.shape[1] == 2:
                return np.apply_along_axis(
                    remap_single_coord_from_inverted_to_standard_y_axis,
                    1, coord)
        elif isinstance(coord, dict):
            return {key: remap_inverted_to_standard_y_axis(value)
                    for key, value in coord.items()}
        return coord

    def map_coord_to_eyelink_space(coord):
        def span(array):
            return max(array) - min(array)
        def remap_single_coord_to_eyelink_space(coord):
            monitor_info = defaults.fetch_monitor_info()
            hor_rez = monitor_info['horizontal_resolution']
            vert_rez = monitor_info['vertical_resolution']
            x_px_range = [-hor_rez * 0.2, hor_rez + hor_rez * 0.2]
            y_px_range = [-vert_rez * 0.2, vert_rez + vert_rez * 0.2]
            return [
                span(x_px_range) * (coord[0] / span(x_px_range)) + min(x_px_range),
                span(y_px_range) * (coord[1] / span(y_px_range)) + min(y_px_range)]
        if not params.get('map_roi_coord_to_eyelink_space', False):
            return coord
        if (isinstance(coord, (tuple, list)) and len(coord) == 2) \
            or (isinstance(coord, np.ndarray) and coord.ndim == 1):
            remapped_coord = remap_single_coord_to_eyelink_space(coord)
            return type(coord)(remapped_coord) \
                if isinstance(coord, (tuple, list)) \
                    else np.array(remapped_coord)
        elif isinstance(coord, np.ndarray) \
            and coord.ndim == 2 \
                and coord.shape[1] == 2:
            return np.apply_along_axis(
                remap_single_coord_to_eyelink_space, 1, coord)
        elif isinstance(coord, dict):
            return {key: map_coord_to_eyelink_space(value)
                    for key, value in coord.items()}
        else:
            raise ValueError(
                "Input must be a 2-element tuple/list or a 2D array with 2 columns")

    def stretch_bounding_box_corners(bb_corner_coord_dict, scale=1.3):
        if isinstance(bb_corner_coord_dict, dict):
            mean_x = sum(point[0] for point in
                         bb_corner_coord_dict.values()) / len(bb_corner_coord_dict)
            mean_y = sum(point[1] for point in 
                        bb_corner_coord_dict.values()) / len(bb_corner_coord_dict)
            shifted_points = {key: (point[0] - mean_x, point[1] - mean_y)
                              for key, point in bb_corner_coord_dict.items()}
            scaled_points = {key: (point[0] * scale, point[1] * scale)
                             for key, point in shifted_points.items()}
            stretched_points = {key: (point[0] + mean_x, point[1] + mean_y)
                             for key, point in scaled_points.items()}
            return stretched_points
        else:
            raise ValueError(
                "Input for 'stretch_from_center_of_mass' must be a dictionary")

    if remapping_type == 'inverted_to_standard_y_axis':
        return remap_inverted_to_standard_y_axis(coord)
    elif remapping_type == 'to_eyelink_space':
        return map_coord_to_eyelink_space(coord)
    elif remapping_type == 'stretch_from_center_of_mass':
        stretch_bounding_box_corners(coord, scale=1.3)
    return coord


def get_bl_and_tr_roi_coords_m1(m1_landmarks, params):
    """
    Calculates the bounding box corners for regions of interest (ROIs) based on M1 landmarks.
    Parameters:
    - m1_landmarks (dict): Dictionary containing M1 landmarks data.
    - map_roi_coord_to_eyelink_space (bool): Flag indicating whether to map coordinates to Eyelink space.
    Returns:
    - bbox_corners (dict): Dictionary with keys 'eye_bbox', 'face_bbox', 'left_obj_bbox', 'right_obj_bbox'
      containing bounding box corners for respective regions.
    """
    # Calculate bounding box corners for each ROI
    eye_bbox = construct_eye_bounding_box( m1_landmarks, params )
    face_bbox = construct_face_bounding_box( m1_landmarks, params )
    left_obj_bbox = construct_object_bounding_box(m1_landmarks, params, 'leftObject')
    right_obj_bbox = construct_object_bounding_box(m1_landmarks, params, 'rightObject')
    return {'eye_bbox': eye_bbox,
        'face_bbox': face_bbox,
        'left_obj_bbox': left_obj_bbox,
        'right_obj_bbox': right_obj_bbox}


def construct_eye_bounding_box(m1_landmarks, params):
    """
    Constructs the bounding box for the eyes.
    Parameters:
    - m1_landmarks (dict): Dictionary containing landmarks for the eyes.
    - params (dict): Parameters dictionary.
        - map_roi_coord_to_eyelink_space (bool): Flag indicating whether to map coordinates to Eyelink space.
    Returns:
    - eye_bb_corners (dict): Dictionary containing eye bounding box coordinates.
    """
    # Extract and remap coordinates for left and right eyes
    left_eye = remap_source_coords(m1_landmarks['eyeOnLeft'][0][0][0],
                                   params, 'inverted_to_standard_y_axis')
    left_eye = remap_source_coords(left_eye, params, 'to_eyelink_space')
    right_eye = remap_source_coords(m1_landmarks['eyeOnRight'][0][0][0],
                                    params, 'inverted_to_standard_y_axis')
    right_eye = remap_source_coords(right_eye, params, 'to_eyelink_space')
    # Validate left_eye and right_eye coordinates
    if not (len(left_eye) == len(right_eye) == 2):
        raise ValueError("Left eye and right eye coordinates should be 2-element tuples, lists, or arrays.")
    # Calculate the center of mass
    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2
    # Calculate inter-eye distance using Euclidean norm
    inter_eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    # Calculate offset
    offset = inter_eye_dist / 2
    # Calculate bounding box corners
    bottom_left = (center_x - 2 * offset, center_y - offset)
    top_right = (center_x + 2 * offset, center_y + offset)
    bbox_dict = {'bottomLeft': bottom_left, 'topRight': top_right}
    return remap_source_coords(bbox_dict, params, 'stretch_from_center_of_mass')


def construct_face_bounding_box(m1_landmarks, params):
    """
    Constructs a bounding box square for the face.
    Parameters:
    - m1_landmarks (dict): Landmarks dictionary.
    - params (dict): Parameters dictionary.
    Returns:
    - bounding_box (dict): Bounding box dictionary containing 'bottomLeft' and 'topRight' corners.
    """
    # Extract and remap coordinates for the face
    face_coords = {key: m1_landmarks[key][0][0][0] for key
                   in ['topLeft', 'topRight', 'bottomLeft', 'bottomRight']}
    face_coords = remap_source_coords(face_coords, 
                                     params, 'inverted_to_standard_y_axis')
    face_coords = remap_source_coords(face_coords,
                                      params, 'to_eyelink_space')
    # Find pairs of corners and calculate distances
    max_distance = 0
    max_distance_corners = None
    for pair in itertools.combinations(face_coords.keys(), 2):
        corner1, corner2 = pair
        distance = np.linalg.norm(np.array(face_coords[corner1])
                                  - np.array(face_coords[corner2]))
        if distance > max_distance:
            max_distance = distance
            max_distance_corners = (corner1, corner2)
    # Check if the points are diagonally opposite
    if not (set(max_distance_corners) == set(['topLeft', 'bottomRight']) or
            set(max_distance_corners) == set(['topRight', 'bottomLeft'])):
        raise ValueError(
            "The points with maximum distance should be diagonally opposite.")
    # Calculate the center of the bounding box
    center_x = np.mean([face_coords[corner][0] for corner in face_coords])
    center_y = np.mean([face_coords[corner][1] for corner in face_coords])
    # Calculate the side length of the bounding box square using the largest x or y distance
    max_x_distance = max(abs(face_coords[corner1][0] - face_coords[corner2][0])
                         for corner1, corner2 in 
                        itertools.combinations(face_coords.keys(), 2))
    max_y_distance = max(abs(face_coords[corner1][1] - face_coords[corner2][1])
                         for corner1, corner2 in
                         itertools.combinations(face_coords.keys(), 2))
    side_length = max(max_x_distance, max_y_distance)
    # Calculate the bottomLeft and topRight corners of the bounding box square
    half_side = side_length / 2
    bottom_left = (center_x - half_side, center_y - half_side)
    top_right = (center_x + half_side, center_y + half_side)
    bbox_dict = {'bottomLeft': bottom_left, 'topRight': top_right}
    return remap_source_coords(bbox_dict, params, 'stretch_from_center_of_mass')


def construct_object_bounding_box(m1_landmarks, params, which_object):
    if which_object == 'leftObject' or which_object == 'rightObject':
        coord = m1_landmarks[which_object][0][0][0]
        bottom_left = remap_source_coords(coord['bottomLeft'][0][0],
                                          params, 'inverted_to_standard_y_axis')
        bottom_left = remap_source_coords(bottom_left, params, 'to_eyelink_space')
        top_right = remap_source_coords(coord['topRight'][0][0],
                                          params, 'inverted_to_standard_y_axis')
        top_right = remap_source_coords(top_right, params, 'to_eyelink_space')
        bbox_dict = {'bottomLeft': bottom_left, 'topRight': top_right}
    else:
        raise ValueError("Input 'which_object' must be a leftObject or rightObject.")
    return remap_source_coords(bbox_dict, params, 'stretch_from_center_of_mass')


def get_filename_flag_info(params):
    """
    Constructs a filename flag based on specified parameters.
    Parameters:
    - params (dict): Dictionary containing parameters.
    Returns:
    - flag_info (str): Filename flag.
    """
    flag_info = ""
    if params.get('map_roi_coord_to_eyelink_space', False):
        flag_info += "_remapped_roi"
    if params.get('map_gaze_pos_coord_to_eyelink_space', False):
        flag_info += "_remapped_gaze"
    return flag_info


def create_timevec(n_samples, sampling_rate):
    """
    Creates a time vector based on the number of samples and sampling rate.
    Parameters:
    - n_samples (int): Number of samples.
    - sampling_rate (float): Sampling rate.
    Returns:
    - timevec (list): Time vector.
    """
    return [i * sampling_rate for i in range(n_samples)]


def find_islands(binary_vec, min_samples=0):
    """
    Finds continuous islands in a binary vector.
    Parameters:
    - binary_vec (array): Binary vector.
    - min_samples (int): Minimum number of samples for an island.
    Returns:
    - islands (array): Array containing start and stop indices of islands.
    """
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
    if island_started and len(binary_vec) - island_start >= min_samples:
        islands.append([island_start, len(binary_vec) - 1])
    return np.array(islands)


def get_duration(start_stop):
    """
    Calculates the duration between start and stop indices.
    Parameters:
    - start_stop (tuple): Tuple containing start and stop indices.
    Returns:
    - duration (int): Duration.
    """
    start, stop = start_stop
    return stop - start


def get_fix_positions(start_stop, positions):
    """
    Retrieves fix positions from a given range of indices.
    Parameters:
    - start_stop (tuple): Tuple containing start and stop indices.
    - positions (array): Array containing positions.
    Returns:
    - fix_positions (array): Array containing fix positions.
    """
    start, stop = start_stop
    return positions[start:stop,:]


def is_inside_roi(coord, bbox_corner_dict):
    # Extract the bounding box corners
    top_right = bbox_corner_dict['topRight']
    bottom_left = bbox_corner_dict['bottomLeft']
    
    # Function to check if a single coordinate is inside the bounding box
    def is_inside_single(coord):
        x, y = coord
        return bottom_left[0] <= x <= top_right[0] and bottom_left[1] <= y <= top_right[1]
    
    # Check if coord is a single coordinate or a list of coordinates
    if isinstance(coord, (list, tuple)) and len(coord) == 2 and isinstance(coord[0], (int, float)):
        # Single coordinate case
        return is_inside_single(coord)
    else:
        # List of coordinates case
        return [is_inside_single(c) for c in coord]
    
    

def is_inside_quadrilateral(point, corners, tolerance=1e-3):
    """
    Checks if a point is inside a quadrilateral.
    Parameters:
    - point (tuple): Point coordinates.
    - corners (dict): Dictionary containing corner coordinates.
    - tolerance (float): Tolerance level for area difference.
    Returns:
    - inside_quad (bool): True if the point is inside the quadrilateral, False otherwise.
    - area_diff (float): Difference in area.
    """
    x, y = point
    x1, y1 = corners['topLeft']
    x2, y2 = corners['topRight']
    x3, y3 = corners['bottomRight']
    x4, y4 = corners['bottomLeft']
    total_area = get_area_using_shoelace_4pts(x1, y1, x2, y2, x3, y3, x4, y4)
    triangle_area_point1 = get_area_using_shoelace_3pts(x, y, x1, y1, x2, y2)
    triangle_area_point2 = get_area_using_shoelace_3pts(x, y, x2, y2, x3, y3)
    triangle_area_point3 = get_area_using_shoelace_3pts(x, y, x3, y3, x4, y4)
    triangle_area_point4 = get_area_using_shoelace_3pts(x, y, x4, y4, x1, y1)
    sum_of_triangles = triangle_area_point1 + triangle_area_point2 + \
        triangle_area_point3 + triangle_area_point4
    area_diff = abs(total_area - sum_of_triangles)
    inside_quad = area_diff < tolerance
    return inside_quad, area_diff


def get_area_using_shoelace_3pts(x1, y1, x2, y2, x3, y3):
    """
    Calculates the area of a triangle using the Shoelace formula.
    Parameters:
    - x1, y1, x2, y2, x3, y3: Coordinates of the triangle vertices.
    Returns:
    - area: The area of the triangle.
    """
    return 0.5 * abs((x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1))


def get_area_using_shoelace_4pts(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Calculates the area of a quadrilateral using the Shoelace formula.
    Parameters:
    - x1, y1, x2, y2, x3, y3, x4, y4: Coordinates of the quadrilateral vertices.
    Returns:
    - area: The area of the quadrilateral.
    """
    total_area = get_area_using_shoelace_3pts(x1, y1, x2, y2, x3, y3) + \
                 get_area_using_shoelace_3pts(x1, y1, x3, y3, x4, y4)
    return total_area


def distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    Parameters:
    - point1 (tuple): First point coordinates.
    - point2 (tuple): Second point coordinates.
    Returns:
    - dist (float): Euclidean distance.
    """
    x1, y1 = point1
    x2, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)










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

