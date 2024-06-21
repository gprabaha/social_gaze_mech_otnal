#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:50:31 2024

@author: pg496
"""


import logging
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
import os
import numpy as np
import ast
import pickle


def make_session_rasters_serial(session_paths, params):
    results = []
    for session_path in session_paths:
        session_raster = generate_session_raster(session_path, params)
        results.append(session_raster)
    return results


def make_session_rasters_parallel(session_paths, params):
    results = []
    with ProcessPoolExecutor() as executor:
        future_to_session = {executor.submit(generate_session_raster, path, params): path for path in session_paths}
        for future in as_completed(future_to_session):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                session_path = future_to_session[future]
                logging.error(f"Session {session_path} generated an exception: {exc}")
    return results


def generate_session_raster(session, labelled_fixations, labelled_spiketimes, params):
    logging.debug(f"Processing session: {session}")
    raster_bin_size = float(params['raster_bin_size'])
    raster_pre_event_time = float(params['raster_pre_event_time'])
    raster_post_event_time = float(params['raster_post_event_time'])
    num_bins = int((raster_pre_event_time + raster_post_event_time) / raster_bin_size)
    session_fixations = labelled_fixations[labelled_fixations['session_name'] == session]
    session_neurons = labelled_spiketimes[labelled_spiketimes['session_name'] == session]
    logging.debug(f"Session fixations shape: {session_fixations.shape}")
    logging.debug(f"Session neurons shape: {session_neurons.shape}")
    if session_fixations.empty or session_neurons.empty:
        logging.warning(f"No data found for session {session}.")
        return None
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(
            process_unit, uuid, session_fixations, session_neurons,
            num_bins, raster_bin_size, raster_pre_event_time, raster_post_event_time): uuid for uuid in session_neurons['uuid'].unique()}
        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logging.error(f"Error processing unit {futures[future]}: {e}")
    if not results:
        logging.warning(f"No results for session {session}.")
        return None
    session_data = pd.concat(results, ignore_index=True)
    session_file_path = os.path.join(params['processed_data_dir'], f"{session}_raster.pkl")
    save_to_pickle(session_data, session_file_path)
    logging.info(f"Saved session data for {session} to {session_file_path}")
    return session_data


def process_unit(uuid, session_fixations, session_neurons, num_bins, raster_bin_size, raster_pre_event_time, raster_post_event_time):
    logging.debug(f"Processing unit: {uuid}")
    neuron_spikes_str = session_neurons[session_neurons['uuid'] == uuid]['spikeS'].values[0]
    
    # Update to use stored numpy arrays or lists directly
    neuron_spikes = np.array(ast.literal_eval(neuron_spikes_str))
    bins = np.arange(-raster_pre_event_time, raster_post_event_time, raster_bin_size)
    results = []
    for _, fixation in session_fixations.iterrows():
        for aligned_to in ['start_time', 'end_time']:
            event_time = float(fixation[aligned_to])
            relevant_spikes = neuron_spikes[(neuron_spikes >= event_time - raster_pre_event_time) & (neuron_spikes < event_time + raster_post_event_time)]
            spike_times = relevant_spikes - event_time
            raster = np.histogram(spike_times, bins=bins)[0]
            raster = raster.astype(int)
            session_data = update_session_data(raster, fixation, session_neurons, uuid, aligned_to)
            results.append(session_data)
    if not results:
        logging.warning(f"No results for unit {uuid}.")
        return None
    return pd.DataFrame(results)


def update_session_data(raster, fixation, session_neurons, uuid, aligned_to):
    session_data = {
        'raster': raster,  # Keep raster as a numpy array
        'category': fixation['category'],
        'session_name': fixation['session_name'],
        'run': fixation['run'],
        'block': fixation['block'],
        'fix_duration': fixation['fix_duration'],
        'mean_x_pos': fixation['mean_x_pos'],
        'mean_y_pos': fixation['mean_y_pos'],
        'fix_roi': fixation['fix_roi'],
        'agent': fixation['agent'],
        'channel': session_neurons[session_neurons['uuid'] == uuid]['channel'].values[0],
        'channel_label': session_neurons[session_neurons['uuid'] == uuid]['channel_label'].values[0],
        'unit_no_within_channel': session_neurons[session_neurons['uuid'] == uuid]['unit_no_within_channel'].values[0],
        'unit_label': session_neurons[session_neurons['uuid'] == uuid]['unit_label'].values[0],
        'uuid': uuid,
        'n_spikes': session_neurons[session_neurons['uuid'] == uuid]['n_spikes'].values[0],
        'region': session_neurons[session_neurons['uuid'] == uuid]['region'].values[0],
        'aligned_to': aligned_to,
        'behavior': 'fixation'
    }
    return session_data


def save_to_pickle(dataframe, filename):
    """
    Function to save the DataFrame to a pickle file.
    Parameters:
    dataframe (pd.DataFrame): DataFrame containing the data to be saved.
    filename (str): Path to the file where data should be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(dataframe, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Data saved to {filename}")


def save_labelled_fixation_rasters(labelled_fixation_rasters, params):
    processed_data_dir = params['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)
    file_path = os.path.join(processed_data_dir, 'labelled_fixation_rasters.pkl')
    # Save DataFrame to a pickle file
    save_to_pickle(labelled_fixation_rasters, file_path)
    logging.info(f"Saved labelled fixation rasters to {file_path}")