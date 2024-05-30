#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:43:33 2024

@author: pg496
"""


import util
import plotter


params = {}
params.update({
    'is_cluster': True,
    'use_parallel': True,
    'remake_labelled_gaze_pos': True,
    'remake_fixations': True,
    'remake_fixation_labels': True,
    'remake_spikeTs': False,
    'remap_source_coord_from_inverted_to_standard_y_axis': True,
    'map_roi_coord_to_eyelink_space': False,
    'map_gaze_pos_coord_to_eyelink_space': True
})


# Determine root data directory based on whether it's running on a cluster or not
root_data_dir, params = util.fetch_root_data_dir(params)
data_source_dir, params = util.fetch_data_source_dir(params)
session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
processed_data_dir, params = util.fetch_processed_data_dir(params)

plotter.plot_gaze_heatmaps_for_conditions(params)

plotter.plot_fixation_heatmaps_for_conditions(params)






