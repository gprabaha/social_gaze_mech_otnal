#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:36:36 2024

@author: prabaha
"""

import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import os
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import ast
import logging
import h5py

import util
import load_data
import process_eyelink
import defaults
import fix
import hpc_cluster

import pdb


### Function to extract meta-information and update params
def extract_and_update_meta_info(params):
    """
    Extracts meta-information from files in session paths and updates the params dictionary.
    Parameters:
    - params (dict): Dictionary containing session paths and other parameters.
    Returns:
    - params (dict): Updated dictionary with meta-information and dose arrays.
    """
    meta_info_list = []
    for session_path in params['session_paths']:
        dose_info = load_data.get_monkey_and_dose_data(session_path)
        if dose_info is not None:
            meta_info = {
                'session_name': os.path.basename(
                    os.path.normpath(session_path))}
            meta_info.update(dose_info)
            runs_info = load_data.get_runs_data(session_path)
            meta_info.update(runs_info)
            meta_info['roi_bb_corners'] = \
                load_data.load_farplane_cal_and_get_bl_and_tr_roi_coords_m1(
                    session_path, params)
            meta_info_list.append(meta_info)
    params['meta_info_list'] = meta_info_list
    otnal_doses = np.array(
        [[meta_info['OT_dose'], meta_info['NAL_dose']]
         for meta_info in meta_info_list], dtype=np.float64)
    params['otnal_doses'] = otnal_doses
    return params



### Function to get unique doses
def get_unique_doses(params):
    """
    Finds unique rows and their indices in the given array.
    Parameters:
    - otnal_doses (ndarray): Input array.
    Returns:
    - unique_rows (ndarray): Unique rows in the input array.
    - indices_for_unique_rows (list): List of lists containing indices for
    each unique row.
    """
    otnal_doses = params['otnal_doses']
    unique_rows = np.unique(otnal_doses, axis=0)
    indices_for_unique_rows = []
    session_category = np.empty(otnal_doses.shape[0])
    session_category[:] = np.nan
    for i, row in enumerate(unique_rows):
        category = i
        indices_for_row = np.where( (otnal_doses == row).all(axis=1))[0]
        session_category[indices_for_row] = category
        indices_for_unique_rows.append(indices_for_row.tolist())
    params.update({'unique_doses': unique_rows,
                   'dose_inds': indices_for_unique_rows,
                   'session_categories': session_category})
    return params

###
def extract_labelled_gaze_positions_m1(params):
    """
    Extracts labelled gaze positions from files associated with unique doses.
    Parameters:
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of tuples containing gaze
    positions and associated metadata.
    """
    processed_data_dir = params['processed_data_dir']
    unique_doses = params.get('unique_doses')
    dose_inds = params.get('dose_inds')
    use_parallel = params.get('use_parallel', True)
    
    def process_index(idx):
        return load_data.get_labelled_gaze_positions_dict_m1(idx, params)
    
    dose_index_pairs = [(dose, idx) for dose, indices_list
                        in zip(unique_doses, dose_inds)
                        for idx in indices_list]
    labelled_gaze_positions_m1 = process_eyelink.process_gaze_positions(
        dose_index_pairs, use_parallel, process_index)
    process_eyelink.save_labelled_gaze_positions(
        processed_data_dir, labelled_gaze_positions_m1, params)
    return labelled_gaze_positions_m1


###
def extract_fixations_with_labels_parallel(labelled_gaze_positions, params):
    """
    Extracts fixations with labels, possibly in parallel.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions
    and associated metadata.
    - params (dict): Dictionary of parameters.
    Returns:
    - all_fixation_labels (pd.DataFrame): DataFrame of labels for fixations.
    """
    print("\nStarting to extract fixations:")
    use_parallel = params.get('use_parallel', True)
    all_fix_timepos, fix_detection_results = fix.extract_or_load_fixations(
        labelled_gaze_positions, params)
    labelled_fixations = fix.generate_fixation_labels(
        fix_detection_results, params, use_parallel)
    return labelled_fixations








### Function to extract saccades with labels
def extract_saccades_with_labels(labelled_gaze_positions, params):
    """
    Extracts saccades with labels.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions and associated metadata.
    - params (dict): Dictionary containing parameters including parallel processing options.
    Returns:
    - labelled_saccades (DataFrame): DataFrame containing saccade information with labels.
    """
    saccade_params = defaults.fetch_default_saccade_pars()
    vel_thresh = saccade_params['vel_thresh']
    min_samples = saccade_params['min_samples']
    smooth_func = saccade_params['smooth_func']
    use_parallel = params.get('use_parallel', False)
    num_sessions = len(labelled_gaze_positions)
    available_cpus = os.cpu_count()
    n_jobs = min(available_cpus, num_sessions)
    sessions_data = [(session_data[0], session_data[1],
                      vel_thresh, min_samples, smooth_func)
                     for session_data in labelled_gaze_positions]
    if use_parallel:
        with tqdm_joblib(tqdm(
                desc="Extracting saccades in parallel",
                total=num_sessions, unit="session")):
            results = Parallel(n_jobs=n_jobs)(
                delayed(extract_saccades_for_session)(session_data)
                for session_data in sessions_data)
    else:
        results = [extract_saccades_for_session(session_data)
                   for session_data in
                   tqdm(sessions_data,
                        desc="Extracting saccades",
                        unit="session")]
    saccades = [s for session_saccades in results for s in session_saccades]
    columns = ["start_time", "end_time", "duration", "trajectory",
               "start_roi", "end_roi", "session_name", "category",
               "run", "block"]
    labelled_saccades = pd.DataFrame(saccades, columns=columns)
    save_saccade_labels(labelled_saccades, params)
    return labelled_saccades


def extract_saccades_for_session(session_data):
    """
    Extracts saccades for a single session.
    Parameters:
    - session_data (tuple): Tuple containing gaze positions, session info, and saccade parameters.
    Returns:
    - session_saccades (list): List of saccades for the session.
    """
    positions, info, vel_thresh, min_samples, smooth_func = session_data
    session_saccades = []
    sampling_rate = info['sampling_rate']
    n_samples = positions.shape[0]
    time_vec = util.create_timevec(n_samples, sampling_rate)
    category = info['category']
    session_name = info['session_name']
    n_runs = info['num_runs']
    for run in range(n_runs):
        run_start = info['startS'][run]
        run_stop = info['stopS'][run]
        run_time = (time_vec > run_start) & (time_vec <= run_stop)
        run_positions = positions[run_time, :]
        run_x = util.px2deg(run_positions[:, 0].T)
        run_y = util.px2deg(run_positions[:, 1].T)
        saccade_start_stops = find_saccades(
            run_x, run_y, sampling_rate, vel_thresh,
            min_samples, smooth_func)
        for start, stop in saccade_start_stops:
            saccade = run_positions[start:stop + 1, :]
            start_time = time_vec[start]
            end_time = time_vec[stop]
            duration = end_time - start_time
            start_roi = determine_roi_of_coord(run_positions[start, :2],
                                      info['roi_bb_corners'])
            end_roi = determine_roi_of_coord(run_positions[stop, :2],
                                    info['roi_bb_corners'])
            block = determine_block(
                start_time, end_time, info['startS'], info['stopS'])
            session_saccades.append(
                [start_time, end_time, duration, saccade,
                 start_roi, end_roi, session_name, category, run, block])
    return session_saccades


def find_saccades(x, y, sr, vel_thresh, min_samples, smooth_func):
    """
    Finds saccades.
    Parameters:
    - x (array-like): x-coordinates of eye movements.
    - y (array-like): y-coordinates of eye movements.
    - sr (float): Sampling rate.
    - vel_thresh (float): Minimum velocity threshold for saccade onset.
    - min_samples (int): Minimum duration of a saccade in samples.
    - smooth_func (function): Function for smoothing input data.
    Returns:
    - start_stops (list): List of start and stop indices of saccades.
    """
    assert x.shape == y.shape
    start_stops = []
    x0 = smooth_func(x)
    y0 = smooth_func(y)
    vx = np.gradient(x0) / sr
    vy = np.gradient(y0) / sr
    vel_norm = np.sqrt(vx ** 2 + vy ** 2)  # Norm of velocity vector
    above_thresh = (vel_norm >= vel_thresh[0]) & (vel_norm <= vel_thresh[1])
    start_stops = util.find_islands(above_thresh, min_samples)
    return start_stops


def determine_roi_of_coord(position, bbox_corners):
    """
    Determines the ROI based on position and bounding box corners.
    Parameters:
    - position (ndarray): Position coordinates.
    - bbox_corners (dict): Dictionary containing bounding boxes of ROIs.
    Returns:
    - roi (str): Detected ROI.
    """
    bounding_boxes = ['eye_bbox', 'face_bbox',
                      'left_obj_bbox', 'right_obj_bbox']
    inside_roi = [util.is_inside_roi(position, bbox_corners[key])
                  for key in bounding_boxes]
    if any(inside_roi):
        if inside_roi[0] and inside_roi[1]:
            return bounding_boxes[0]
        return bounding_boxes[inside_roi.index(True)]
    return 'out_of_roi'


def determine_block(start_time, end_time, startS, stopS):
    """
    Determines the block for a saccade based on start and stop times.
    Parameters:
    - start_time (float): Start time of the saccade.
    - end_time (float): End time of the saccade.
    - startS (list): List of start indices of runs.
    - stopS (list): List of stop indices of runs.
    Returns:
    - block (str): Detected block.
    """
    if start_time < startS[0] or end_time > stopS[-1]:
        return 'discard'
    for i, (run_start, run_stop) in enumerate(zip(startS, stopS), start=1):
        if start_time >= run_start and end_time <= run_stop:
            return 'mon_down'
        elif i < len(startS) and end_time <= startS[i]:
            return 'mon_up'
    return 'discard'


def save_saccade_labels(labelled_saccades, params):
    """
    Saves the labelled saccades to a specified directory.
    Parameters:
    - labelled_saccades (DataFrame): DataFrame containing saccade information with labels.
    - params (dict): Dictionary containing parameters including the save directory.
    """
    processed_data_dir = params['processed_data_dir']
    flag_info = util.get_filename_flag_info(params)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    file_path = os.path.join(
        processed_data_dir, f'labelled_saccades{flag_info}.csv')
    labelled_saccades.to_csv(file_path, index=False)
    print(f"Saccade labels saved to {file_path}")


def extract_spiketimes_for_all_sessions(params):
    processed_data_dir = params.get('processed_data_dir')
    session_paths = params.get('session_paths')
    is_parallel = params.get('use_parallel', True)
    spikeTs_labels = []

    if is_parallel:
        # Process sessions in parallel with tqdm progress bar
        results = Parallel(n_jobs=-1)(delayed(
            load_data.get_spiketimes_and_labels_for_one_session)(
                session_path, processed_data_dir)
            for session_path in tqdm(
                session_paths, desc='Loading spiketimes'))
        # Iterate over results and concatenate
        for labelled_spiketimes in tqdm(results, desc='Concatenating results'):
            spikeTs_labels.append(labelled_spiketimes)
    else:
        # Process sessions sequentially
        for session_path in session_paths:
            labelled_spiketimes = load_data.get_spiketimes_and_labels_for_one_session(
                session_path, processed_data_dir)
            spikeTs_labels.append(labelled_spiketimes)

    # Concatenate label dataframes
    if spikeTs_labels:
        all_labels = pd.concat(spikeTs_labels, ignore_index=True)
    else:
        all_labels = pd.DataFrame()

    # Construct flag_info based on params
    flag_info = util.get_filename_flag_info(params)
    
    # Define the HDF5 file path
    h5_file_path = os.path.join(processed_data_dir, f'spike_labels{flag_info}.h5')

    # Save DataFrame to HDF5
    save_spiketimes_to_hdf5(all_labels, h5_file_path)
    print(f"All labelled spiketimes saved to {h5_file_path}")

    return all_labels

def save_spiketimes_to_hdf5(df, file_path):
    """
    Save the dataframe to an HDF5 file with `spikeS` and `spikeMs` as variable-length datasets.
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    file_path (str): The file path to save the HDF5 file.
    """
    with h5py.File(file_path, 'w') as hf:
        spikeS_group = hf.create_group('spikeS')
        spikeMs_group = hf.create_group('spikeMs')
        labels_group = hf.create_group('labels')
        for index, row in df.iterrows():
            spikeS_data = row['spikeS']
            spikeMs_data = row['spikeMs']
            # Create variable-length datasets for spikeS and spikeMs
            spikeS_group.create_dataset(str(index), data=spikeS_data, dtype=h5py.vlen_dtype(float))
            spikeMs_group.create_dataset(str(index), data=spikeMs_data, dtype=h5py.vlen_dtype(float))
        # Save the remaining columns (labels)
        for column in df.columns:
            if column not in ['spikeS', 'spikeMs']:
                labels_group.create_dataset(column, data=df[column].values)
    logging.info(f"Data successfully saved to {file_path}")



# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def process_sessions_serial(session_paths, params):
    results = []
    for session_path in session_paths:
        session_raster = generate_session_raster(session_path, params)
        results.append(session_raster)
    return results


def process_sessions_parallel(session_paths, params):
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


def extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, params):
    session_names = [os.path.basename(session_path) for session_path in session_paths]
    logging.debug(f"Session names extracted from paths: {session_names}")
    results = []
    
    if params.get('remake_raster', False):
        if params.get('submit_separate_jobs_for_session_raster', True):
            #job_file_path = hpc_cluster.generate_job_file(session_paths)
            #hpc_cluster.submit_job_array(job_file_path)
            # Wait for job completion is handled within submit_job_array
            session_files = [os.path.join(params['processed_data_dir'], f"{session}_raster.pkl") for session in session_names]
            for session_file in session_files:
                try:
                    session_data = load_data.load_session_raster_data(session_file)
                    results.append(session_data)
                except FileNotFoundError as e:
                    logging.error(e)
                    continue
            if not results:
                logging.error("No results to concatenate.")
                raise ValueError("No objects to concatenate")
            labelled_fixation_rasters = pd.concat(results, ignore_index=True)
        else:
            if params.get('use_parallel', False):
                results = process_sessions_parallel(session_paths, params)
            else:
                results = process_sessions_serial(session_paths, params)
            
            if not results:
                logging.error("No results to concatenate.")
                raise ValueError("No objects to concatenate")
            
            labelled_fixation_rasters = pd.concat(results, ignore_index=True)
    else:
        session_files = []
        for session in session_names:
            session_file_path = os.path.join(params['processed_data_dir'], f"{session}_raster.pkl")
            try:
                logging.info(f"Loading existing data for session {session} from {session_file_path}")
                session_data = load_data.load_session_raster_data(session_file_path)
                session_files.append(session_data)
            except FileNotFoundError as e:
                logging.error(e)
                continue
        if not session_files:
            logging.error("No files to concatenate.")
            raise ValueError("No objects to concatenate")
        labelled_fixation_rasters = pd.concat(session_files, ignore_index=True)
    
    save_labelled_fixation_rasters(labelled_fixation_rasters, params)
    return labelled_fixation_rasters



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
            raster = (raster > 0).astype(int)
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











