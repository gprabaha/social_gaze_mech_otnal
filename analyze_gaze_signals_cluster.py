#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:41:20 2024

@author: pg496
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

import pdb

import load_data
import util
import filter_behavior

'''
No fixation is now being detected inside the left object bounding box. This is
most probably the position data points have not been transformed to be remapped
between the edges of the bounds of the eyetracker rect
'''

# Read parameters from environment variables
map_roi_coord_to_eyelink_space = os.getenv('map_roi_coord_to_eyelink_space', 'False') == 'True'
map_gaze_pos_coord_to_eyelink_space = os.getenv('map_gaze_pos_coord_to_eyelink_space', 'False') == 'True'

params = {}
params.update({
    'is_cluster': True,
    'use_parallel': True,
    'remake_labelled_gaze_pos': True,
    'remake_fixations': True,
    'remake_fixation_labels': True,
    'remake_spikeTs': False,
    'map_roi_coord_to_eyelink_space': map_roi_coord_to_eyelink_space,
    'map_gaze_pos_coord_to_eyelink_space': map_gaze_pos_coord_to_eyelink_space
})

# Determine root data directory based on whether it's running on a cluster or not
root_data_dir = util.get_root_data_dir(params)
params.update({'root_data_dir': root_data_dir})

session_paths = util.get_subfolders(params)
params.update({'session_paths': session_paths})

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
    fixations_m1, fix_timepos_m1, fixation_labels_m1 = \
        filter_behavior.extract_fixations_with_labels_parallel(
            labelled_gaze_positions_m1, params)  # The first file has funky session stop times
else:
    fixations_m1, fix_timepos_m1 = load_data.load_m1_fixations(params)
    all_fixation_labels = load_data.load_m1_fixation_labels(params)

if params.get('remake_spikeTs'):
    spikeTs_s, spikeTs_ms, spikeTs_labels = filter_behavior.extract_spiketimes_for_all_sessions(params)
else:
    spikeTs_s, spikeTs_ms, spikeTs_labels = load_data.load_processed_spike_data(params)

# ROIs fixated on
rois_with_fixatins = fixation_labels_m1['fix_roi'].unique()
print(f"Remap ROI: {params['map_roi_coord_to_eyelink_space']}\n")
print(f"Remap gaze positions: {params['map_gaze_pos_coord_to_eyelink_space']}\n")
print(f'Rois with fixations detected in them are:\n{rois_with_fixatins}\n')

# ROI Indices
face_roi_bool_inds = fixation_labels_m1['fix_roi'] == 'face_bbox'
eye_roi_bool_inds = fixation_labels_m1['fix_roi'] == 'eye_bbox'
left_obj_roi_bool_inds = fixation_labels_m1['fix_roi'] == 'left_obj_bbox'
right_obj_roi_bool_inds = fixation_labels_m1['fix_roi'] == 'right_obj_bbox'

# Agent Indices
lynch_bool_inds = fixation_labels_m1['agent'] == 'Lynch'
tarantino_bool_inds = fixation_labels_m1['agent'] == 'Tarantino'

# Monitor up or down
mon_up_bool_inds = fixation_labels_m1['block'] == 'mon_up'
mon_down_bool_inds = fixation_labels_m1['block'] == 'mon_down'

all_sessions = fixation_labels_m1['session_name'].unique()

for session in all_sessions:
    session_fix_bool_inds = fixation_labels_m1['session_name'] == session
    session_unit_indices = np.where(spikeTs_labels['session_name'] == session)[0]
    for unit_index in session_unit_indices:
        unit_spikeTs_s = spikeTs_s[unit_index]


"""
# Find saccades
saccades_m1, saccade_labels_m1 = filter_behavior.extract_saccades_with_labels(labelled_gaze_positions_m1)
np.save(os.path.join(root_data_dir, 'saccades_m1.npz'), saccades_m1)
np.save(os.path.join(root_data_dir, 'saccade_labels_m1.npz'), saccade_labels_m1)
"""


