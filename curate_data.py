#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:36:36 2024

@author: prabaha
"""

import numpy as np
from tqdm import tqdm
import pickle
import os
import pandas as pd
import logging
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

import util
import load_data
import eyelink
import fix_and_saccades
from raster import RasterManager
from hpc_cluster import HPCCluster

import pdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    labelled_gaze_positions_m1 = eyelink.process_gaze_positions(
        dose_index_pairs, use_parallel, process_index)
    eyelink.save_labelled_gaze_positions(
        processed_data_dir, labelled_gaze_positions_m1, params)
    return labelled_gaze_positions_m1


def generate_toy_gazepos_data(labelled_gaze_positions, params):
    third_session_data = labelled_gaze_positions[2]
    # Ensure third_session_data[0] is a NumPy array with shape (N, 2)
    if not isinstance(third_session_data[0], np.ndarray) or third_session_data[0].shape[1] != 2:
        raise ValueError("Expected third_session_data[0] to be a NumPy array with shape (N, 2)")
    array_data = third_session_data[0]
    N = array_data.shape[0]
    # Determine the length of the continuous segment to extract
    h = N // 20
    # Choose a random starting index i such that the segment [i to i + h] is within bounds
    i = np.random.randint(0, N - h)
    # Extract the continuous segment
    toy_data = (array_data[i:i + h], third_session_data[1])
    # Save the toy_data in the processed_data_dir specified in params
    processed_data_dir = params['processed_data_dir']
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    toy_data_path = os.path.join(processed_data_dir, 'toy_data.pkl')
    with open(toy_data_path, 'wb') as f:
        pickle.dump(toy_data, f)
    return [toy_data]



###
def extract_fixations_and_saccades_with_labels(labelled_gaze_positions, params):
    """
    Extracts fixations and saccades with labels, possibly in parallel.
    Parameters:
    - labelled_gaze_positions (list): List of tuples containing gaze positions
    and associated metadata.
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_fixations (pd.DataFrame): DataFrame of labels for fixations.
    - labelled_saccades (pd.DataFrame): DataFrame containing saccade information with labels.
    """
    print("\nStarting to extract fixations and saccades:")
    use_parallel = params.get('use_parallel', True)

    # Extract fixations and saccades
    all_fix_df, all_saccades_df = fix_and_saccades.extract_or_load_fixations_and_saccades(labelled_gaze_positions, params)
    # labelled_fixations = fix_and_saccades.generate_fixation_labels(fix_detection_results, params, use_parallel)
    # saccades = [s for session_saccades in saccade_detection_results for s in session_saccades]
    # columns = ["start_time", "end_time", "duration", "trajectory", "start_roi", "end_roi", "session_name", "category", "run", "block"]
    # labelled_saccades = pd.DataFrame(saccades, columns=columns)
    # fix_and_saccades.save_saccade_labels(labelled_saccades, params)

    return all_fix_df, all_saccades_df


###
def extract_spiketimes_for_all_sessions(params):
    processed_data_dir = params.get('processed_data_dir')
    session_paths = params.get('session_paths')
    is_parallel = params.get('use_parallel', True)
    spikeTs_labels = []
    if is_parallel:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            # Submit tasks and collect futures
            futures = {
                executor.submit(
                    load_data.get_spiketimes_and_labels_for_one_session, 
                    session_path, 
                    processed_data_dir
                ): session_path for session_path in session_paths
            }
            # Process futures as they complete with tqdm progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc='Loading spiketimes'):
                try:
                    result = future.result()
                    spikeTs_labels.append(result)
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
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
    h5_file_path = os.path.join(processed_data_dir, f'labelled_spiketimes{flag_info}.h5')
    # Save DataFrame to HDF5
    save_spiketimes_to_hdf5(all_labels, h5_file_path)
    print(f"All labelled spiketimes saved to {h5_file_path}")
    return all_labels



def save_spiketimes_to_hdf5(labelled_spiketimes, file_path):
    with h5py.File(file_path, 'w') as hf:
        spikeS_group = hf.create_group('spikeS')
        spikeMs_group = hf.create_group('spikeMs')
        labels_group = hf.create_group('labels')
        for index, row in labelled_spiketimes.iterrows():
            spikeS_data = np.array(row['spikeS'], dtype=float).tolist()
            spikeMs_data = np.array(row['spikeMs'], dtype=float).tolist()
            spikeS_group.create_dataset(str(index), data=spikeS_data)
            spikeMs_group.create_dataset(str(index), data=spikeMs_data)
        for label in ['session_name', 'channel', 'channel_label', 'unit_no_within_channel', 'unit_label', 'uuid', 'n_spikes', 'region']:
            # Convert strings to bytes
            data_as_bytes = [str(item).encode('utf-8') for item in labelled_spiketimes[label]]
            labels_group.create_dataset(label, data=np.array(data_as_bytes))





def extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, params):
    session_names = [os.path.basename(session_path) for session_path in session_paths]
    logging.debug(f"Session names extracted from paths: {session_names}")
    results = []
    raster_manager = RasterManager(params)
    
    if params.get('remake_raster', False):
        if params.get('submit_separate_jobs_for_sessions', True):
            hpc_cluster = HPCCluster(params)
            job_file_path = hpc_cluster.generate_job_file(session_paths)
            hpc_cluster.submit_job_array(job_file_path)
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
                results = raster_manager.make_session_rasters_parallel(session_paths, labelled_fixations, labelled_spiketimes)
            else:
                results = raster_manager.make_session_rasters_serial(session_paths, labelled_fixations, labelled_spiketimes)
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
    
    raster_manager.save_labelled_fixation_rasters(labelled_fixation_rasters)
    return labelled_fixation_rasters
















