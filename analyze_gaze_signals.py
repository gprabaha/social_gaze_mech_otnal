#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""


import os
import pandas as pd
import glob
import argparse
import logging
import subprocess
import datetime


import pdb

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def generate_job_file(session_paths):
    job_file_path = '/gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/raster_joblist.txt'
    os.makedirs('job_scripts', exist_ok=True)
    with open(job_file_path, 'w') as file:
        for session_path in session_paths:
            command = f"module load miniconda; conda init bash; conda activate nn_gpu; python analyze_gaze_signals.py --session {session_path}"
            file.write(command + "\n")
    return job_file_path


def submit_job_array(job_file_path):
    try:
        # Ensure the directory paths are correct and use absolute paths
        output_dir = '/gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/'
        job_script_path = os.path.join(output_dir, 'dsq-joblist_raster.sh')

        # Run the command to generate the job script
        subprocess.run(
            f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} -o {output_dir} --status-dir {output_dir} --cpus-per-task 4 --mem-per-cpu 16g -t 02:00:00 --mail-type FAIL',
            shell=True, check=True, executable='/bin/bash'
        )
        logging.info("Successfully generated the dSQ job script")

        # Check if the job script file exists
        if not os.path.isfile(job_script_path):
            logging.error(f"No job script found at {job_script_path}.")
            return

        logging.info(f"Using dSQ job script: {job_script_path}")

        # Submit the job script with sbatch and ensure output is directed to the job_scripts directory
        subprocess.run(
            f'sbatch --output={output_dir}/%x_%A_%a.out --error={output_dir}/%x_%A_%a.err {job_script_path}',
            shell=True, check=True, executable='/bin/bash'
        )
        logging.info(f"Successfully submitted jobs using sbatch for script {job_script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during job submission process: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gaze signals for a specific session")
    parser.add_argument('--session', type=str, help='Session ID to process')
    parser.add_argument('--generate_jobs', action='store_true', help='Generate job file and submit job array')
    args = parser.parse_args()

    # Load necessary data and parameters
    import util
    import curate_data
    import load_data
    import plotter

    root_data_dir, params = util.fetch_root_data_dir(params)
    data_source_dir, params = util.fetch_data_source_dir(params)
    session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
    processed_data_dir, params = util.fetch_processed_data_dir(params)

    if params.get('remake_labelled_gaze_pos'):
        params = curate_data.extract_and_update_meta_info(params)
        params = curate_data.get_unique_doses(params)
        labelled_gaze_positions_m1 = curate_data.extract_labelled_gaze_positions_m1(params)
    else:
        labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(params)

    if params.get('remake_fixations') or params.get('remake_fixation_labels'):
        labelled_fixations = curate_data.extract_fixations_with_labels_parallel(labelled_gaze_positions_m1, params)
    else:
        labelled_fixations = load_data.load_m1_fixation_labels(params)

    if params.get('remake_saccades'):
        labelled_saccades_m1 = curate_data.extract_saccades_with_labels(labelled_gaze_positions_m1, params)
    else:
        labelled_saccades_m1 = load_data.load_saccade_labels(params)

    if params.get('remake_spikeTs'):
        labelled_spiketimes = curate_data.extract_spiketimes_for_all_sessions(params)
    else:
        labelled_spiketimes = load_data.load_processed_spiketimes(params)

    # Log shapes of the loaded data
    logging.debug(f"First few rows of labelled fixations: \n{labelled_fixations.head()}")
    logging.debug(f"First few rows of labelled spiketimes: \n{labelled_spiketimes.head()}")
    logging.debug(f"Labelled fixations shape: {labelled_fixations.shape}")
    logging.debug(f"Labelled spiketimes shape: {labelled_spiketimes.shape}")

    if args.generate_jobs:
        job_file_path = generate_job_file(session_paths)
        submit_job_array(job_file_path)
    elif args.session:
        # Process a single session
        session_name = os.path.basename(os.path.normpath(args.session))
        labelled_fixations = labelled_fixations[labelled_fixations['session_name'] == session_name]
        labelled_spiketimes = labelled_spiketimes[labelled_spiketimes['session_name'] == session_name]
        curate_data.generate_session_raster(session_name, labelled_fixations, labelled_spiketimes, params)
    else:
        # Process all sessions
        if params.get('remake_raster'):
            labeled_fixation_rasters = curate_data.extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, params)
        else:
            labeled_fixation_rasters = load_data.load_labelled_fixation_rasters(params)

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