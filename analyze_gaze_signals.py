#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""


import logging
import util
import curate_data
import load_data
import response_comp

def flush_variable(variable_name, global_vars, logger):
    if variable_name in global_vars:
        del global_vars[variable_name]
        logger.info(f"Flushed variable: {variable_name}")

def get_or_load_variable(variable_name, load_function, compute_function, params, global_vars, logger):
    # Detailed debug information
    variable_exists = variable_name in global_vars and global_vars[variable_name] is not None

    if variable_exists:
        logger.info(f"Variable '{variable_name}' found in globals and is not None.")
    else:
        logger.info(f"Variable '{variable_name}' NOT found in globals or is None.")

    if params.get('use_existing_variables', False) and variable_exists:
        logger.info(f"Using existing variable: {variable_name}")
        return global_vars[variable_name]

    if params.get('flush_before_reload'):
        flush_variable(variable_name, global_vars, logger)

    if params.get(f'remake_{variable_name}', False) or not variable_exists:
        if params.get(f'remake_{variable_name}', False):
            logger.info(f"Recomputing variable: {variable_name}")
            global_vars[variable_name] = compute_function(params)
        else:
            logger.info(f"Loading variable: {variable_name}")
            global_vars[variable_name] = load_function(params)

    logger.info(f"Variable '{variable_name}' set in globals.")
    return global_vars[variable_name]

def main(params, labelled_gaze_positions_m1=None, labelled_fixations=None, labelled_saccades_m1=None, labelled_spiketimes=None, labelled_fixation_rasters=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)  # Set the logging level to ERROR
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    

    global_vars = globals()

    # Load necessary data and parameters
    params.update({
        'is_cluster': True,
        'use_parallel': False,
        'remake_labelled_gaze_pos': False,
        'remake_fixations': False,
        'remake_fixation_labels': False,
        'remake_saccades': False,
        'remake_spikeTs': False,
        'remake_raster': False,
        'make_plots': False,
        'recalculate_unit_ROI_responses': True,
        'replot_face/eye_vs_obj_violins': True,
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
        'raster_post_event_time': 0.5,
        'flush_before_reload': False,
        'use_existing_variables': True,
        'reload_existing_unit_roi_comp_stats': False  # New flag to reload existing stats
    })

    root_data_dir, params = util.fetch_root_data_dir(params)
    data_source_dir, params = util.fetch_data_source_dir(params)
    session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
    processed_data_dir, params = util.fetch_processed_data_dir(params)

    # Labelled Gaze Positions
    logger.debug("Loading or computing labelled_gaze_positions_m1")
    labelled_gaze_positions_m1 = get_or_load_variable(
        'labelled_gaze_positions_m1',
        load_data.load_labelled_gaze_positions,
        lambda p: curate_data.extract_labelled_gaze_positions_m1(curate_data.get_unique_doses(curate_data.extract_and_update_meta_info(p))),
        params,
        global_vars,
        logger
    )
    logger.info("Loaded or set labelled_gaze_positions_m1")

    # Labelled Fixations
    logger.debug("Loading or computing labelled_fixations")
    labelled_fixations = get_or_load_variable(
        'labelled_fixations',
        load_data.load_m1_fixation_labels,
        lambda p: curate_data.extract_fixations_with_labels_parallel(labelled_gaze_positions_m1, p),
        params,
        global_vars,
        logger
    )
    logger.info("Loaded or set labelled_fixations")

    # Labelled Saccades
    logger.debug("Loading or computing labelled_saccades_m1")
    labelled_saccades_m1 = get_or_load_variable(
        'labelled_saccades_m1',
        load_data.load_saccade_labels,
        lambda p: curate_data.extract_saccades_with_labels(labelled_gaze_positions_m1, p),
        params,
        global_vars,
        logger
    )
    logger.info("Loaded or set labelled_saccades_m1")

    # Labelled Spiketimes
    logger.debug("Loading or computing labelled_spiketimes")
    labelled_spiketimes = get_or_load_variable(
        'labelled_spiketimes',
        load_data.load_processed_spiketimes,
        curate_data.extract_spiketimes_for_all_sessions,
        params,
        global_vars,
        logger
    )
    logger.info("Loaded or set labelled_spiketimes")

    # Labelled Fixation Rasters
    logger.debug("Loading or computing labelled_fixation_rasters")
    labelled_fixation_rasters = get_or_load_variable(
        'labelled_fixation_rasters',
        load_data.load_labelled_fixation_rasters,
        lambda p: curate_data.extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, p),
        params,
        global_vars,
        logger
    )
    logger.info("Loaded or set labelled_fixation_rasters")

    if params.get('replot_face/eye_vs_obj_violins'):
        response_comp.compute_pre_and_post_fixation_response_to_roi_for_each_unit(labelled_fixation_rasters, params)



global params, labelled_gaze_positions_m1, labelled_fixations, labelled_saccades_m1, labelled_spiketimes, labelled_fixation_rasters

params = util.get_params()
labelled_gaze_positions_m1 = labelled_gaze_positions_m1 if 'labelled_gaze_positions_m1' in globals() else None
labelled_fixations = labelled_fixations if 'labelled_fixations' in globals() else None
labelled_saccades_m1 = labelled_saccades_m1 if 'labelled_saccades_m1' in globals() else None
labelled_spiketimes = labelled_spiketimes if 'labelled_spiketimes' in globals() else None
labelled_fixation_rasters = labelled_fixation_rasters if 'labelled_fixation_rasters' in globals() else None

main(params, labelled_gaze_positions_m1, labelled_fixations, labelled_saccades_m1, labelled_spiketimes, labelled_fixation_rasters)





























