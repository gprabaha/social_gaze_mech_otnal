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

from multiprocessing import Pool

import pdb

logger = logging.getLogger(__name__)


def extract_labelled_gaze_positions_m1(params):
    """
    Extracts labelled gaze positions from files associated with unique doses.
    Parameters:
    - params (dict): Dictionary of parameters.
    Returns:
    - labelled_gaze_positions_m1 (list): List of tuples containing gaze
    positions and associated metadata.
    - params (dict): Updated dictionary of parameters.
    """
    processed_data_dir = params['processed_data_dir']
    use_parallel = params.get('use_parallel', True)
    params = extract_and_update_meta_info(params)
    params = get_unique_doses(params)
    unique_doses = params.get('unique_doses')
    dose_inds = params.get('dose_inds')
    dose_index_pairs = [(dose, idx) for dose, indices_list
                        in zip(unique_doses, dose_inds)
                        for idx in indices_list]
    labelled_gaze_positions_m1, params = eyelink.process_gaze_positions(
        dose_index_pairs, use_parallel, params)
    eyelink.save_labelled_gaze_positions(
        processed_data_dir, labelled_gaze_positions_m1, params)
    return labelled_gaze_positions_m1, params


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
    # Extract fixations and saccades
    if params.get('remake_labelled_fixations_m1', False) or params.get('remake_labelled_fixations_m1', False):
        all_fix_df, all_saccades_df = fix_and_saccades.extract_all_fixations_and_saccades_from_labelled_gaze_positions(labelled_gaze_positions, params)
    else:
        all_fix_df, all_saccades_df = load_data.load_m1_labelled_fixations_and_saccades(params)
    if params.get('remake_combined_behav_m1', True):
        combined_behav_df = combine_behaviors_in_temporal_order(params, all_fix_df, all_saccades_df)
    return all_fix_df, all_saccades_df, combined_behav_df


def sort_behavioral_event_dataframes_in_session(df):
    """
    Sorts the behavioral event DataFrame by 'start_index' and 'end_index'.
    Input:
    - df (pd.DataFrame): DataFrame containing behavioral event data.
    Output:
    - pd.DataFrame: Sorted DataFrame.
    Usage:
    sorted_df = sort_behavioral_event_dataframes_in_session(df)
    """
    return df.sort_values(by=['start_index', 'end_index'])


def check_clashes(session_df):
    """
    Checks for time window clashes in the session DataFrame.
    Input:
    - session_df (pd.DataFrame): DataFrame containing sorted session data.
    Output:
    - list: List of clashes with tuples containing session name, previous
        row index, current row index, previous end index, and current start index.
    Usage:
    clashes = check_clashes(session_df)
    """
    clashes = []
    for i in range(1, len(session_df)):
        prev_end_index = session_df.iloc[i-1]['end_index']
        curr_start_index = session_df.iloc[i]['start_index']
        if curr_start_index < prev_end_index:
            clashes.append((session_df.iloc[i]['session_name'], i-1, i, prev_end_index, curr_start_index))
    return clashes


def combine_behaviors_in_temporal_order(params, *dataframes):
    """
    Combines fixation and saccade DataFrames into a single DataFrame in temporal order.
    Input:
    - params (dict): Dictionary containing parameters.
    - dataframes (pd.DataFrame): Two DataFrames, one for fixations and one for saccades.
    Output:
    - pd.DataFrame: Combined and sorted DataFrame with an added 'event_type' column.
    Usage:
    combined_df = combine_behaviors_in_temporal_order(params, fixation_df, saccade_df)
    """
    use_parallel = params.get('use_parallel', False)
    num_cpus = params.get('num_cpus', 1)
    if len(dataframes) != 2:
        raise ValueError("Exactly two DataFrames must be provided, one for fixations and one for saccades.")
    for df, event_type in zip(dataframes, ['fixation', 'saccade']):
        df['event_type'] = event_type
    combined_df = pd.concat(dataframes, ignore_index=True)
    unique_sessions = combined_df['session_name'].unique()
    logger.info("Starting sorting of sessions.")
    if use_parallel:
        with Pool(num_cpus) as pool:
            sorted_dfs = list(tqdm(pool.imap(sort_behavioral_event_dataframes_in_session, 
                                             [combined_df[combined_df['session_name'] == session] for session in unique_sessions]), 
                                   total=len(unique_sessions), desc="Sorting session behav dfs"))
            pool.close()
            pool.join()
    else:
        sorted_dfs = [sort_behavioral_event_dataframes_in_session(combined_df[combined_df['session_name'] == session])
                      for session in unique_sessions]
    final_sorted_df = pd.concat(sorted_dfs, ignore_index=True)
    logger.info("Finished sorting sessions. Checking for time window clashes.")
    if use_parallel:
        with Pool(num_cpus) as pool:
            results = list(tqdm(pool.imap(check_clashes, 
                                          [final_sorted_df[final_sorted_df['session_name'] == session] for session in unique_sessions]), 
                                total=len(unique_sessions), desc="Checking time window clashes within events"))
            pool.close()
            pool.join()
    else:
        results = [check_clashes(final_sorted_df[final_sorted_df['session_name'] == session])
                   for session in unique_sessions]
    clashes = [clash for result in results for clash in result]
    if clashes:
        logger.info("Time window clashes found:")
        for clash in clashes:
            logger.info(f"Session: {clash[0]}, Row {clash[1]} ends at index {clash[3]}, but Row {clash[2]} starts at index {clash[4]}")
    else:
        logger.info("No time window clashes found.")
    mon_down_events = final_sorted_df[final_sorted_df['block'] == 'mon_down'].head(50)
    logger.info(f"First 50 events in the 'mon_down' condition:\n{mon_down_events}")
    save_combined_df_to_csv(params, final_sorted_df, 'combined_gaze_behav_m1.csv')
    return final_sorted_df


def save_combined_df_to_csv(params, df, filename):
    """
    Saves the provided DataFrame to a CSV file in the directory specified by params['processed_data_dir'].
    Args:
    - params (dict): A dictionary containing parameters including 'processed_data_dir'.
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): The name of the file to save the DataFrame as.
    Returns:
    - None
    """
    processed_data_dir = params['processed_data_dir']
    # Ensure the directory exists
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
        logger.info(f"Created directory: {processed_data_dir}")
    # Define the full path for the output file
    output_path = os.path.join(processed_data_dir, filename)
    # Save the DataFrame as a CSV file
    df.to_csv(output_path, index=False)
    logger.info(f"DataFrame saved to: {output_path}")


def isolate_events_within_attention_frame(dataframe, gaze_data, use_parallel=False):
    """
    Isolates events within the frame of attention for all sessions.
    Utilizes parallel processing to handle each session separately if use_parallel is True.
    Args:
    - dataframe (pd.DataFrame): DataFrame containing all events.
    - gaze_data (list): List of gaze data containing session information and gaze positions.
    - use_parallel (bool, optional): Flag to determine whether to use parallel processing. Default is False.
    Returns:
    - pd.DataFrame: DataFrame containing events within the frame of attention.
    """
    if use_parallel:
        with Pool() as pool:
            results = list(tqdm(pool.starmap(isolate_events_for_session, [(dataframe, session_data) for session_data in gaze_data]),
                                total=len(gaze_data),
                                desc="Isolating events within frame of attention"))
            pool.close()
            pool.join()
    else:
        results = [isolate_events_for_session(dataframe, session_data)
                   for session_data in tqdm(
                       gaze_data, desc="Isolating events within frame of attention")]
    events_within_frame = pd.concat(results, ignore_index=True)
    return events_within_frame


def isolate_events_for_session(dataframe, session_data):
    """
    Isolates events within the frame of attention for a single session.
    Args:
    - dataframe (pd.DataFrame): DataFrame containing all events.
    - session_data (tuple): Tuple containing gaze data for a single session.
    Returns:
    - pd.DataFrame: DataFrame containing events within the frame of attention for the session.
    """
    events_within_frame = []
    session_name = session_data[1]['session_name']
    bboxes = session_data[1]['roi_bb_corners']
    frame = util.define_frame_of_attention(bboxes)
    session_events = dataframe[dataframe['session_name'] == session_name]
    gaze_positions = session_data[0]  # x, y positions array
    for index, row in session_events.iterrows():
        start_index = row['start_index']
        end_index = row['end_index']
        event_type = row['event_type']
        mean_position_str = row['mean_position']
        # Convert mean_position to numpy array
        mean_position = util.convert_to_array(mean_position_str)
        # Fetch the positions from gaze data
        start_position = gaze_positions[start_index]
        end_position = gaze_positions[end_index]
        try:
            if (util.is_within_frame(start_position, frame) or 
                util.is_within_frame(end_position, frame) or 
                (event_type == 'fixation' and util.is_within_frame(mean_position, frame))):
                events_within_frame.append(row)
        except Exception as e:
            print(f"Error processing row {index}: {row}")
            print(f"Exception: {e}")
    return pd.DataFrame(events_within_frame)




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
                    params,
                    session_path
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
                params, session_path)
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
        # Store unit_validity as boolean
        unit_validity = labelled_spiketimes['unit_validity'].astype(bool)
        labels_group.create_dataset('unit_validity', data=unit_validity)





def extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, params):
    session_names = [os.path.basename(session_path) for session_path in session_paths]
    logger.debug(f"Session names extracted from paths: {session_names}")
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
                    logger.error(e)
                    continue
            if not results:
                logger.error("No results to concatenate.")
                raise ValueError("No objects to concatenate")
            labelled_fixation_rasters = pd.concat(results, ignore_index=True)
        else:
            if params.get('use_parallel', False):
                results = raster_manager.make_session_rasters_parallel(session_paths, labelled_fixations, labelled_spiketimes)
            else:
                results = raster_manager.make_session_rasters_serial(session_paths, labelled_fixations, labelled_spiketimes)
            if not results:
                logger.error("No results to concatenate.")
                raise ValueError("No objects to concatenate")
            labelled_fixation_rasters = pd.concat(results, ignore_index=True)
    else:
        session_files = []
        for session in session_names:
            session_file_path = os.path.join(params['processed_data_dir'], f"{session}_raster.pkl")
            try:
                logger.info(f"Loading existing data for session {session} from {session_file_path}")
                session_data = load_data.load_session_raster_data(session_file_path)
                session_files.append(session_data)
            except FileNotFoundError as e:
                logger.error(e)
                continue
        if not session_files:
            logger.error("No files to concatenate.")
            raise ValueError("No objects to concatenate")
        labelled_fixation_rasters = pd.concat(session_files, ignore_index=True)
    raster_manager.save_labelled_fixation_rasters(labelled_fixation_rasters)
    return labelled_fixation_rasters
















