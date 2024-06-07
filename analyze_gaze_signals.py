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
import curate_data
import plotter


params = {}
params.update({
    'is_cluster': True,
    'use_parallel': True,
    'remake_labelled_gaze_pos': False,
    'remake_fixations': False,
    'remake_fixation_labels': False,
    'remake_saccades': False,
    'remake_spikeTs': False,
    'make_plots': False,
    'remap_source_coord_from_inverted_to_standard_y_axis': True, # !!Important
    'map_roi_coord_to_eyelink_space': False,
    'map_gaze_pos_coord_to_eyelink_space': True,
    'export_plots_to_local_folder': False,
    'inter_eye_dist_denom_for_eye_bbox_offset': 2,
    'offset_multiples_in_x_dir': 3,
    'offset_multiples_in_y_dir': 1.5,
    'bbox_expansion_factor': 1.3
})

"""

- Use the start and end times of various ROI fixations to during the
monitor-up and monitor down blocks as long as they are not discards. Then
use the timepoints to go to the neural spiketimes for that session to compare
firing rates corresponding to looks within for each of the bboxes between
the monitor-up and monitor-down condition

- Start looking at CEBRA stuff

"""


# Determine root data directory based on whether it's running on a cluster or not
root_data_dir, params = util.fetch_root_data_dir(params)
data_source_dir, params = util.fetch_data_source_dir(params)
session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
processed_data_dir, params = util.fetch_processed_data_dir(params)


if params.get('remake_labelled_gaze_pos'):
    params = curate_data.extract_and_update_meta_info(params)
    params = curate_data.get_unique_doses(params)
    labelled_gaze_positions_m1 = \
        curate_data.extract_labelled_gaze_positions_m1(params)
else:
    labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(params)


if params.get('remake_fixations') or params.get('remake_fixation_labels'):
    all_fixation_labels = curate_data.extract_fixations_with_labels_parallel(
        labelled_gaze_positions_m1, params)  # The first file has funky session stop times
else:
    all_fixation_labels = load_data.load_m1_fixation_labels(params)


if params.get('remake_saccades'):
    labelled_saccades_m1 = curate_data.extract_saccades_with_labels(
        labelled_gaze_positions_m1, params)
else:
    labelled_saccades_m1 = load_data.load_saccade_labels(params)


if params.get('remake_spikeTs'):
    labelled_spiketimes = curate_data.extract_spiketimes_for_all_sessions(params)
else:
    labelled_spiketimes = load_data.load_processed_spiketimes(params)

if params.get('make_plots'):
    plotter.plot_fixation_proportions_for_diff_conditions(params)
    plotter.plot_gaze_heatmaps(params)
    plotter.plot_fixation_heatmaps(params)




"""
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