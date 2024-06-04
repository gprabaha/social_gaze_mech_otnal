#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

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
    'use_parallel': False,
    'remake_labelled_gaze_pos': True,
    'remake_fixations': True,
    'remake_fixation_labels': True,
    'remake_spikeTs': False,
    'remap_source_coord_from_inverted_to_standard_y_axis': True,
    'map_roi_coord_to_eyelink_space': False,
    'map_gaze_pos_coord_to_eyelink_space': True,
    'export_plots_to_local_folder': False,
    'inter_eye_dist_denom_for_eye_bbox_offset': 2,
    'offset_multiples_in_x_dir': 3,
    'offset_multiples_in_y_dir': 1,
    'bbox_expansion_factor': 1.5
    
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


plotter.plot_fixation_proportions_for_diff_conditions(params)
plotter.plot_gaze_heatmaps(params)
plotter.plot_fixation_heatmaps(params)


'''
if params.get('remake_fixations') or params.get('remake_fixation_labels'):
    fixations_m1, fix_timepos_m1, fixation_labels_m1 = \
        filter_behavior.extract_fixations_with_labels_parallel(
            labelled_gaze_positions_m1, params)  # The first file has funky session stop times
else:
    fixations_m1, fix_timepos_m1 = load_data.load_m1_fixations(params)
    fixation_labels_m1 = load_data.load_m1_fixation_labels(params)

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

plotter.plot_fixation_proportions_for_diff_conditions(fixation_labels_m1, params)


for session in all_sessions:
    session_fix_bool_inds = fixation_labels_m1['session_name'] == session
    session_unit_indices = np.where(spikeTs_labels['session_name'] == session)[0]
    for unit_index in session_unit_indices:
        unit_spikeTs_s = spikeTs_s[unit_index]
'''






"""
# Find saccades
saccades_m1, saccade_labels_m1 = filter_behavior.extract_saccades_with_labels(labelled_gaze_positions_m1)
np.save(os.path.join(root_data_dir, 'saccades_m1.npz'), saccades_m1)
np.save(os.path.join(root_data_dir, 'saccade_labels_m1.npz'), saccade_labels_m1)

# for each neuron see eye vs obj and also central fix (in interval) vs obj

saccade_lengths_m1 = [saccade.shape[0] for saccade in saccades_m1]
# Plot the histogram
plt.hist(saccade_lengths_m1, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Number of Samples')
plt.ylabel('Frequency')
plt.title('Histogram of Saccade Lengths')
plt.grid(True)
plt.show()


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

max_length = max(len(seq) for seq in saccades)
padded_saccades = pad_sequences(saccades, maxlen=max_length, padding='post', dtype='float32')

# Convert saccade_labels to one-hot encoding
one_hot_labels = [to_categorical(label[0], num_classes=4) for label in saccade_labels]

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout, BatchNormalization
import multiprocessing

# Set TensorFlow to use all available CPU cores
num_cores = multiprocessing.cpu_count()
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=num_cores,
    inter_op_parallelism_threads=num_cores,
    allow_soft_placement=True,
    device_count={'CPU': num_cores})
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Define LSTM model with dropout and renormalization layer
model = Sequential()
model.add(LSTM(units=64, input_shape=(max_length, 2), return_sequences=True))
model.add(Dropout(0.3))  # Add dropout layer with 30% dropout rate
model.add(LSTM(units=64))  # Add another LSTM layer
model.add(Dropout(0.3))  # Add dropout layer with 30% dropout rate
model.add(BatchNormalization())  # Add renormalization layer
model.add(Dense(4, activation='softmax'))  # Output layer with 4 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_saccades, np.array(one_hot_labels), epochs=10, batch_size=32, validation_split=0.2)
"""