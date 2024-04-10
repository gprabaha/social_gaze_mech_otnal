#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""

from filter_behavior import *
from util import *


# Determine root data directory based on whether it's running on a cluster or not
is_cluster = False
root_data_dir = get_root_data_dir(is_cluster)

# Get subfolders within the root data directory
session_paths = get_subfolders(root_data_dir)

# Extract meta-information from session paths
meta_info_list = extract_meta_info(session_paths)

# Extract OT and NAL doses from meta-information and convert to numpy array
otnal_doses = np.array([[meta_info['OT_dose'], meta_info['NAL_dose']] for meta_info in meta_info_list], dtype=np.float64)

# Find unique doses and their indices
unique_doses, dose_inds, session_categories = get_unique_doses(otnal_doses)

# Removing the nans
# unique_doses = unique_doses[:-1]
# dose_inds = dose_inds[:-1]

labelled_gaze_positions = extract_labelled_gaze_positions(unique_doses, dose_inds, meta_info_list, session_paths, session_categories)

# Have to write this function for nn training
saccades_with_labels = extract_saccades_with_labels(labelled_gaze_positions)

