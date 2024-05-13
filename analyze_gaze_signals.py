#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

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


"""
All fixations are out of bounds right now which cannot be correct. Check out what is going on
"""

# Remember to remove outliers from the position data
# Take the center of the monitor for tarantino for inter-run interval and check the fixations there compared to face

# Determine root data directory based on whether it's running on a cluster or not
is_cluster = True
use_parallel = True
remake_labelled_gaze_pos = True
remake_fixations = True
reload_labelled_pos = False

root_data_dir = load_data.get_root_data_dir(is_cluster)
if reload_labelled_pos:
    with open(os.path.join(root_data_dir, 'labelled_gaze_positions_m1.pkl'), 'rb') as f:
        labelled_gaze_positions_m1 = pickle.load(f)
elif 'labelled_gaze_positions_m1' in globals():
    print("labelled_gaze_positions_m1 is already loaded")

# Get subfolders within the root data directory
session_paths = load_data.get_subfolders(root_data_dir)
# Extract meta-information from session paths
meta_info_list = filter_behavior.extract_meta_info(session_paths)
# Extract OT and NAL doses from meta-information and convert to numpy array
otnal_doses = np.array([[meta_info['OT_dose'], meta_info['NAL_dose']] for meta_info in meta_info_list], dtype=np.float64)
# Find unique doses and their indices
unique_doses, dose_inds, session_categories = filter_behavior.get_unique_doses(otnal_doses)

if remake_labelled_gaze_pos:
    labelled_gaze_positions_m1 = filter_behavior.extract_labelled_gaze_positions_m1(
        root_data_dir, unique_doses, dose_inds, meta_info_list, session_paths, session_categories)
if remake_fixations:
    if not reload_labelled_pos:
        labelled_gaze_positions_m1 = filter_behavior.extract_labelled_gaze_positions_m1(
            root_data_dir, unique_doses, dose_inds, meta_info_list, session_paths, session_categories)
    fixations_m1, fixation_labels_m1 = filter_behavior.extract_fixations_with_labels_parallel(
        labelled_gaze_positions_m1, root_data_dir, use_parallel)  # The first file has funky session stop times
    np.save(os.path.join(root_data_dir, 'fixations_m1.npy'), fixations_m1)
    fixation_labels_m1.to_csv(os.path.join(root_data_dir, 'fixation_labels_m1.csv'), index=False)
else:
    fixations_m1 = np.load(os.path.join(root_data_dir, 'fixations_m1.npy'))
    fixation_labels_m1 = pd.read_csv(os.path.join(root_data_dir, 'fixation_labels_m1.csv'))


# if re_extract_spike_ts
spikeTs_s, spikeTs_ms, spikeTs_labels = filter_behavior.extract_spiketimes_for_all_sessions(root_data_dir, session_paths, use_parallel)





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