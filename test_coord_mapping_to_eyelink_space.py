#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:11:35 2024

@author: pg496
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import load_data
import util
import filter_behavior
import plotter

# Read parameters from environment variables
map_roi_coord_to_eyelink_space = os.getenv(
    'map_roi_coord_to_eyelink_space', 'False') == 'True'
map_gaze_pos_coord_to_eyelink_space = os.getenv(
    'map_gaze_pos_coord_to_eyelink_space', 'False') == 'True'

params = {}
params.update({
    'is_cluster': True,
    'use_parallel': True,
    'remake_labelled_gaze_pos': True,
    'remake_fixations': True,
    'remake_fixation_labels': True,
    'remake_spikeTs': False,
    'remap_source_coord_from_inverted_to_standard_y_axis': True,
    'map_roi_coord_to_eyelink_space': map_roi_coord_to_eyelink_space,
    'map_gaze_pos_coord_to_eyelink_space': map_gaze_pos_coord_to_eyelink_space,
    'export_plots_to_local_folder': False
})


# Determine root data directory based on whether it's running on a cluster or not
root_data_dir, params = util.fetch_root_data_dir(params)
data_source_dir, params = util.fetch_data_source_dir(params)
session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
processed_data_dir, params = util.fetch_processed_data_dir(params)


if params.get('remake_labelled_gaze_pos'):
    meta_info_list = filter_behavior.extract_meta_info(params)
    params.update({'meta_info_list': meta_info_list})
    otnal_doses = np.array([[meta_info['OT_dose'], meta_info['NAL_dose']]
                            for meta_info in meta_info_list], dtype=np.float64)
    params.update({'otnal_doses': otnal_doses})
    params = filter_behavior.get_unique_doses(params)
    labelled_gaze_positions_m1 = \
        filter_behavior.extract_labelled_gaze_positions_m1(params)
else:
    labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(params)


if params.get('remake_fixations') or params.get('remake_fixation_labels'):
    all_fixation_labels = filter_behavior.extract_fixations_with_labels_parallel(
        labelled_gaze_positions_m1, params)  # The first file has funky session stop times
else:
    all_fixation_labels = load_data.load_m1_fixation_labels(params)





