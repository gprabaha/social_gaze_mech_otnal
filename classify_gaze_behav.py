#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""

from util import *
from filter_behavior import *

import matplotlib.pyplot as plt

# Determine root data directory based on whether it's running on a cluster or not
is_cluster = True
root_data_dir = get_root_data_dir(is_cluster)
# Get subfolders within the root data directory
session_paths = get_subfolders(root_data_dir)
# Extract meta-information from session paths
meta_info_list = extract_meta_info(session_paths)
# Extract OT and NAL doses from meta-information and convert to numpy array
otnal_doses = np.array([[meta_info['OT_dose'], meta_info['NAL_dose']] for meta_info in meta_info_list], dtype=np.float64)
# Find unique doses and their indices
unique_doses, dose_inds, session_categories = get_unique_doses(otnal_doses)
labelled_gaze_positions = extract_labelled_gaze_positions(unique_doses, dose_inds, meta_info_list, session_paths, session_categories)
# Have to write this function for nn training
saccades, saccade_labels = extract_saccades_with_labels(labelled_gaze_positions)

saccade_lengths = [saccade.shape[0] for saccade in saccades]
# Plot the histogram
plt.hist(saccade_lengths, bins=50, color='skyblue', edgecolor='black')
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

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout, BatchNormalization

# Define LSTM model with dropout and renormalization layer
model = Sequential()
model.add(LSTM(units=64, input_shape=(max_length, 2)))
model.add(Dropout(0.3))  # Add dropout layer with 30% dropout rate
model.add(BatchNormalization())  # Add renormalization layer
model.add(Dense(4, activation='softmax'))  # Output layer with 4 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_saccades, np.array(one_hot_labels), epochs=10, batch_size=32, validation_split=0.2)
