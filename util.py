#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:50:13 2024

@author: prabaha
"""

import os
import numpy as np
from math import degrees, atan2, sqrt
from datetime import datetime
import itertools
import re

import defaults

import pdb


def get_params():
    params = {
        'is_cluster': True,
        'use_parallel': True,
        'remake_labelled_gaze_pos': False,
        'remake_fixations': False,
        'remake_fixation_labels': False,
        'remake_saccades': False,
        'remake_spikeTs': False,
        'remake_raster': True,
        'make_plots': False,
        'remap_source_coord_from_inverted_to_standard_y_axis': True,
        'map_roi_coord_to_eyelink_space': False,
        'map_gaze_pos_coord_to_eyelink_space': True,
        'export_plots_to_local_folder': False,
        'inter_eye_dist_denom_for_eye_bbox_offset': 2,
        'offset_multiples_in_x_dir': 3,
        'offset_multiples_in_y_dir': 1.5,
        'bbox_expansion_factor': 1.3,
        'raster_bin_size': 0.001,  # in seconds
        'raster_pre_event_time': 0.5,
        'raster_post_event_time': 0.5
    }
    return params

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


def add_date_dir_to_path(path):
    # Get the current date as a string in YYYYMMDD format
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(path, date_str)


def remap_source_coords(coord, params, remapping_type, scale=None):
    """
    Remap source coordinates based on the specified remapping type.
    Parameters:
    coord (various): The coordinate(s) to be remapped. Can be a tuple, list, numpy array, or dictionary.
    params (dict): Dictionary of parameters containing flags and additional settings for remapping.
    remapping_type (str): Type of remapping to apply. Options are 'inverted_to_standard_y_axis', 'to_eyelink_space', and 'stretch_from_center_of_mass'.
    scale (float, optional): Scaling factor for 'stretch_from_center_of_mass' remapping type. Defaults to None.
    Returns:
    various: Remapped coordinates in the same format as the input.
    """

    def remap_inverted_to_standard_y_axis(coord):
        """
        Remap coordinates from an inverted y-axis to a standard y-axis.
        Parameters:
        coord (various): The coordinate(s) to be remapped.
        Returns:
        various: Remapped coordinates.
        """
        def remap_single_coord_from_inverted_to_standard_y_axis(coord):
            # Convert coordinate to numpy array and invert the y-axis
            coord = np.array(coord, dtype=np.int16)
            return (coord[0], -coord[1])
        # Handle different input types (tuple, list, numpy array, dictionary)
        if isinstance(coord, (list, tuple)):
            if len(coord) == 2 and all(isinstance(i, (int, float)) for i in coord):
                remapped_coord = remap_single_coord_from_inverted_to_standard_y_axis(coord)
                return type(coord)(remapped_coord)
            elif all(isinstance(i, (list, tuple, np.ndarray)) and len(i) == 2 for i in coord):
                return [remap_single_coord_from_inverted_to_standard_y_axis(c) for c in coord]
        elif isinstance(coord, np.ndarray):
            if coord.ndim == 1 and coord.shape[0] == 2:
                return np.array(remap_single_coord_from_inverted_to_standard_y_axis(coord))
            elif coord.ndim == 2 and coord.shape[1] == 2:
                return np.apply_along_axis(remap_single_coord_from_inverted_to_standard_y_axis, 1, coord)
        elif isinstance(coord, dict):
            return {key: remap_inverted_to_standard_y_axis(value) for key, value in coord.items()}
        return coord

    def map_coord_to_eyelink_space(coord):
        """
        Remap coordinates to EyeLink space.
        Parameters:
        coord (various): The coordinate(s) to be remapped.
        Returns:
        various: Remapped coordinates.
        """
        def span(array):
            return max(array) - min(array)
        def remap_single_coord_to_eyelink_space(coord):
            # Fetch monitor information and calculate pixel ranges
            monitor_info = defaults.fetch_monitor_info()
            hor_rez = monitor_info['horizontal_resolution']
            vert_rez = monitor_info['vertical_resolution']
            x_px_range = [-hor_rez * 0.2, hor_rez + hor_rez * 0.2]
            y_px_range = [-vert_rez * 0.2, vert_rez + vert_rez * 0.2]
            # Convert coordinate to numpy array and remap to EyeLink space
            coord = np.array(coord, dtype=np.int16)
            return [
                span(x_px_range) * (coord[0] / span(x_px_range)) + min(x_px_range),
                span(y_px_range) * (coord[1] / span(y_px_range)) + min(y_px_range)
            ]
        # Handle different input types (tuple, list, numpy array, dictionary)
        if (isinstance(coord, (tuple, list)) and len(coord) == 2) or (isinstance(coord, np.ndarray) and coord.ndim == 1):
            remapped_coord = remap_single_coord_to_eyelink_space(coord)
            return type(coord)(remapped_coord) if isinstance(coord, (tuple, list)) else np.array(remapped_coord)
        elif isinstance(coord, np.ndarray) and coord.ndim == 2 and coord.shape[1] == 2:
            return np.apply_along_axis(remap_single_coord_to_eyelink_space, 1, coord)
        elif isinstance(coord, dict):
            return {key: map_coord_to_eyelink_space(value) for key, value in coord.items()}
        else:
            raise ValueError("Input must be a 2-element tuple/list or a 2D array with 2 columns")

    def stretch_bounding_box_corners(bb_corner_coord_dict, scale=1.3):
        """
        Stretch bounding box corners from the center of mass.
        Parameters:
        bb_corner_coord_dict (dict): Dictionary of bounding box corner coordinates.
        scale (float): Scaling factor for stretching. Default is 1.3.
        Returns:
        dict: Stretched bounding box coordinates.
        """
        if isinstance(bb_corner_coord_dict, dict):
            # Calculate the center of mass
            mean_x = sum(point[0] for point in bb_corner_coord_dict.values()) / len(bb_corner_coord_dict)
            mean_y = sum(point[1] for point in bb_corner_coord_dict.values()) / len(bb_corner_coord_dict)
            # Shift points to the origin, scale them, and shift back to the center of mass
            shifted_points = {key: (point[0] - mean_x, point[1] - mean_y) for key, point in bb_corner_coord_dict.items()}
            scaled_points = {key: (point[0] * scale, point[1] * scale) for key, point in shifted_points.items()}
            stretched_points = {key: (point[0] + mean_x, point[1] + mean_y) for key, point in scaled_points.items()}
            return stretched_points
        else:
            raise ValueError("Input for 'stretch_from_center_of_mass' must be a dictionary")
    # Determine the remapping type and apply the appropriate remapping function
    if remapping_type == 'inverted_to_standard_y_axis':
        return remap_inverted_to_standard_y_axis(coord)
    elif remapping_type == 'to_eyelink_space':
        return map_coord_to_eyelink_space(coord)
    elif remapping_type == 'stretch_from_center_of_mass':
        # Use the provided scale or default to the value in params
        if scale is None:
            scale = params.get('bbox_expansion_factor', 1.3)
        return stretch_bounding_box_corners(coord, scale=scale)
    return coord


def get_bl_and_tr_roi_coords_m1(m1_landmarks, params):
    """
    Calculates the bounding box corners for regions of interest (ROIs) based on M1 landmarks.
    Parameters:
    - m1_landmarks (dict): Dictionary containing M1 landmarks data.
    - params (dict): Dictionary containing parameters including 'session_name'.
    Returns:
    - bbox_corners (dict): Dictionary with keys 'eye_bbox', 'face_bbox', 'left_obj_bbox', 'right_obj_bbox'
      containing bounding box corners for respective regions.
    """
    # Calculate bounding box corners for each ROI
    eye_bbox = construct_eye_bounding_box(m1_landmarks, params)
    face_bbox = construct_face_bounding_box(m1_landmarks, params)
    left_obj_bbox = construct_object_bounding_box(m1_landmarks, params, 'leftObject')
    right_obj_bbox = construct_object_bounding_box(m1_landmarks, params, 'rightObject')
    m1_landmarks_dict = extract_landmarks(m1_landmarks)
    return {
        'eye_bbox': eye_bbox,
        'face_bbox': face_bbox,
        'left_obj_bbox': left_obj_bbox,
        'right_obj_bbox': right_obj_bbox,
        'landmarks_dict': m1_landmarks_dict
    }


def extract_landmarks(landmarks_array):
    # Predefined list of expected labels
    expected_labels = ['topLeft', 'bottomLeft', 'topRight', 'bottomRight', 'eyeOnLeft', 'eyeOnRight', 'mouth', 'leftObject', 'rightObject']
    # Fetch the available keys from the dtype of the array
    keys = landmarks_array.dtype.names
    # Check if the keys match the expected labels
    if set(keys) != set(expected_labels):
        raise ValueError(f"Mismatch in labels. Expected: {expected_labels}, Found: {keys}")
    # Initialize the result dictionary
    landmark_dict = {}
    # Loop over each key and extract the corresponding coordinates
    for key in keys:
        if key in ['leftObject', 'rightObject']:
            # For leftObject and rightObject, extract all four corners
            landmark_dict[key] = {
                'topLeft': landmarks_array[key][0][0][0]['topLeft'][0][0],
                'topRight': landmarks_array[key][0][0][0]['topRight'][0][0],
                'bottomLeft': landmarks_array[key][0][0][0]['bottomLeft'][0][0],
                'bottomRight': landmarks_array[key][0][0][0]['bottomRight'][0][0]
            }
        else:
            try:
                # Fetch the coordinates, assuming the same structure
                landmark_dict[key] = landmarks_array[key][0][0][0]
            except IndexError:
                landmark_dict[key] = []
    return landmark_dict


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
    if params.get('map_roi_coord_to_eyelink_space', False):
        left_eye = remap_source_coords(left_eye, params, 'to_eyelink_space')
    right_eye = remap_source_coords(m1_landmarks['eyeOnRight'][0][0][0],
                                    params, 'inverted_to_standard_y_axis')
    if params.get('map_roi_coord_to_eyelink_space', False):
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
    offset = inter_eye_dist / params['inter_eye_dist_denom_for_eye_bbox_offset']
    # Calculate bounding box corners
    multiple_in_x_dir = params['offset_multiples_in_x_dir']
    multiple_in_y_dir = params['offset_multiples_in_y_dir']
    bottom_left = (center_x - multiple_in_x_dir * offset,
                   center_y - multiple_in_y_dir * offset)
    top_right = (center_x + multiple_in_x_dir * offset,
                 center_y + multiple_in_y_dir * offset)
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
    if params.get('map_roi_coord_to_eyelink_space', False):
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
    max_x_distance = abs(face_coords[max_distance_corners[0]][0]
                         - face_coords[max_distance_corners[1]][0])
    max_y_distance = abs(face_coords[max_distance_corners[0]][1]
                         - face_coords[max_distance_corners[1]][1])
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
        if params.get('map_roi_coord_to_eyelink_space', False):
            bottom_left = remap_source_coords(bottom_left, params, 'to_eyelink_space')
        top_right = remap_source_coords(coord['topRight'][0][0],
                                          params, 'inverted_to_standard_y_axis')
        if params.get('map_roi_coord_to_eyelink_space', False):
            top_right = remap_source_coords(top_right, params, 'to_eyelink_space')
        bbox_dict = {'bottomLeft': bottom_left, 'topRight': top_right}
    else:
        raise ValueError("Input 'which_object' must be a leftObject or rightObject.")
    return remap_source_coords(bbox_dict, params, 'stretch_from_center_of_mass')


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
    """
    Check if coordinates are inside the ROI defined by bounding box corners.
    Parameters:
    - coord (array-like): Single coordinate or array of coordinates.
    - bbox_corner_dict (dict): Dictionary with 'topRight' and 'bottomLeft' corners of the bounding box.
    Returns:
    - bool or list of bool: Whether each coordinate is inside the bounding box.
    """
    # Extract the bounding box corners
    top_right = bbox_corner_dict['topRight']
    bottom_left = bbox_corner_dict['bottomLeft']

    # Function to check if a single coordinate is inside the bounding box
    def is_inside_single(x, y):
        return bottom_left[0] <= x <= top_right[0] and bottom_left[1] <= y <= top_right[1]

    if isinstance(coord, np.ndarray):
        if coord.ndim == 1 and coord.size == 2:
            # Single coordinate case
            return is_inside_single(coord[0], coord[1])
        elif coord.ndim == 2 and coord.shape[1] == 2:
            # Array of coordinates case
            return np.array([is_inside_single(c[0], c[1]) for c in coord])
    elif isinstance(coord, (list, tuple)) and len(coord) == 2 and isinstance(coord[0], (int, float)):
        # Single coordinate case for list or tuple
        return is_inside_single(coord[0], coord[1])
    else:
        # List of coordinates case
        return [is_inside_single(c[0], c[1]) for c in coord]


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
    x2, y2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def define_frame_of_attention(bboxes):
    """
    Define the frame of attention based on the bounding boxes of left and right objects.
    Parameters:
    - bboxes (dict): Dictionary containing bounding boxes with 'bottomLeft' and 'topRight' corners for 'left_obj_bbox' and 'right_obj_bbox'.
    Returns:
    - dict: Dictionary with 'bottomLeft' and 'topRight' corners defining the frame of attention.
    """
    left_bbox_center = np.mean([bboxes['left_obj_bbox']['bottomLeft'], bboxes['left_obj_bbox']['topRight']], axis=0)
    right_bbox_center = np.mean([bboxes['right_obj_bbox']['bottomLeft'], bboxes['right_obj_bbox']['topRight']], axis=0)
    center_point = (left_bbox_center + right_bbox_center) / 2
    bbox_distance = np.linalg.norm(left_bbox_center - right_bbox_center)
    # Calculate the frame boundaries
    left_boundary = center_point[0] - 1.2 * bbox_distance
    right_boundary = center_point[0] + 1.2 * bbox_distance
    mean_height = np.mean([bboxes['left_obj_bbox']['topRight'][1] - bboxes['left_obj_bbox']['bottomLeft'][1],
                           bboxes['right_obj_bbox']['topRight'][1] - bboxes['right_obj_bbox']['bottomLeft'][1]])
    top_boundary = center_point[1] + 1.2 * mean_height
    bottom_boundary = center_point[1] - 2.5 * mean_height
    return {'bottomLeft': (left_boundary, bottom_boundary), 'topRight': (right_boundary, top_boundary)}


def is_within_frame(mean_position, frame):
    try:
        # Log the values for debugging
        bottom_left = frame['bottomLeft']
        top_right = frame['topRight']
        if mean_position.size != 2:
            raise ValueError(f"mean_position does not have exactly two elements: {mean_position}")
        x, y = mean_position
        return bottom_left[0] <= x <= top_right[0] and bottom_left[1] <= y <= top_right[1]
    except Exception as e:
        print(f"Error in is_within_frame: {e}, mean_position: {mean_position}, frame: {frame}")
        raise  # Re-raise the exception to maintain the original error behavior


def convert_to_array(position_str):
    try:
        if isinstance(position_str, np.ndarray):
            return position_str  # If it's already an ndarray, return it as is
        # Remove newline characters and extra spaces
        position_str = position_str.replace('\n', ' ').strip()
        # Use regular expressions to extract numbers, handling both 2D and 1D arrays, including scientific notation
        position_list = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', position_str)
        # Convert to numpy array of floats
        array = np.array(position_list, dtype=float)
        # Check if the array length is greater than 2, indicating a 2D array
        if array.size > 2 and array.size % 2 == 0:
            array = array.reshape(-1, 2)
        return array
    except Exception as e:
        print(f"Error converting position string to array: {e}")
        return None  # or raise an appropriate exception



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













