#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:11:35 2024

@author: pg496
"""

import matplotlib.pyplot as plt
import numpy as np

import load_data
import util
import filter_behavior
import plotter


params = {}
params.update({
    'is_cluster': True,
    'use_parallel': True,
    'remake_labelled_gaze_pos': True,
    'remake_fixations': False,
    'remake_fixation_labels': False,
    'remake_spikeTs': False,
    'remap_source_coord_from_inverted_to_standard_y_axis': True,
    'map_roi_coord_to_eyelink_space': True,
    'map_gaze_pos_coord_to_eyelink_space': False
})

# Determine root data directory based on whether it's running on a cluster or not
root_data_dir, params = util.fetch_root_data_dir(params)
data_source_dir, params = util.fetch_data_source_dir(params)
session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
processed_data_dir, params = util.fetch_processed_data_dir(params)

for remap_gaze_value in [True, False]:
    for map_roi_value in [True, False]:
        params['map_roi_coord_to_eyelink_space'] = map_roi_value
        params['map_gaze_pos_coord_to_eyelink_space'] = remap_gaze_value
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
