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
from joblib import Parallel, delayed
import pandas as pd
import logging
import h5py

import util
import load_data
import eyelink
import defaults
import fix
import saccade
import hpc_cluster
import raster

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
    labelled_gaze_positions_m1 = eyelink.process_gaze_positions(
        dose_index_pairs, use_parallel, process_index)
    eyelink.save_labelled_gaze_positions(
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
                delayed(saccade.extract_saccades_for_session)(session_data)
                for session_data in sessions_data)
    else:
        results = [saccade.extract_saccades_for_session(session_data)
                   for session_data in
                   tqdm(sessions_data,
                        desc="Extracting saccades",
                        unit="session")]
    saccades = [s for session_saccades in results for s in session_saccades]
    columns = ["start_time", "end_time", "duration", "trajectory",
               "start_roi", "end_roi", "session_name", "category",
               "run", "block"]
    labelled_saccades = pd.DataFrame(saccades, columns=columns)
    saccade.save_saccade_labels(labelled_saccades, params)
    return labelled_saccades


###
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
def extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, params):
    session_names = [os.path.basename(session_path) for session_path in session_paths]
    logging.debug(f"Session names extracted from paths: {session_names}")
    results = []
    if params.get('remake_raster', False):
        if params.get('submit_separate_jobs_for_session_raster', True):
            job_file_path = hpc_cluster.generate_job_file(session_paths)
            hpc_cluster.submit_job_array(job_file_path)
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
                results = raster.make_session_rasters_parallel(session_paths, params)
            else:
                results = raster.make_session_rasters_serial(session_paths, params)
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
    raster.save_labelled_fixation_rasters(labelled_fixation_rasters, params)
    return labelled_fixation_rasters















