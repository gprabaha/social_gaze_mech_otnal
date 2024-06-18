#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""


import logging
import pdb

logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Only show warnings and errors
# Set up logging
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)  # Only show warnings and errors
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


import util
import curate_data
import load_data
import response_comp


def flush_variable(variable_name, globals_dict):
    if variable_name in globals_dict:
        del globals_dict[variable_name]


if __name__ == "__main__":
    # Load necessary data and parameters
    params = util.get_params()
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
        'recalculate_unit_ROI_responses': False,
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
        'flush_before_reload': True
    })

    root_data_dir, params = util.fetch_root_data_dir(params)
    data_source_dir, params = util.fetch_data_source_dir(params)
    session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
    processed_data_dir, params = util.fetch_processed_data_dir(params)

    if params.get('flush_before_reload'):
        flush_variable('labelled_gaze_positions_m1', globals())
    if params.get('remake_labelled_gaze_pos'):
        params = curate_data.extract_and_update_meta_info(params)
        params = curate_data.get_unique_doses(params)
        labelled_gaze_positions_m1 = curate_data.extract_labelled_gaze_positions_m1(params)
    else:
        labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(params)

    if params.get('flush_before_reload'):
        flush_variable('labelled_fixations', globals())
    if params.get('remake_fixations') or params.get('remake_fixation_labels'):
        labelled_fixations = curate_data.extract_fixations_with_labels_parallel(labelled_gaze_positions_m1, params)
    else:
        labelled_fixations = load_data.load_m1_fixation_labels(params)

    if params.get('flush_before_reload'):
        flush_variable('labelled_saccades_m1', globals())
    if params.get('remake_saccades'):
        labelled_saccades_m1 = curate_data.extract_saccades_with_labels(labelled_gaze_positions_m1, params)
    else:
        labelled_saccades_m1 = load_data.load_saccade_labels(params)

    if params.get('flush_before_reload'):
        flush_variable('labelled_spiketimes', globals())
    if params.get('remake_spikeTs'):
        labelled_spiketimes = curate_data.extract_spiketimes_for_all_sessions(params)
    else:
        labelled_spiketimes = load_data.load_processed_spiketimes(params)

    if params.get('flush_before_reload'):
        flush_variable('labelled_fixation_rasters', globals())
    if params.get('remake_raster'):
        labelled_fixation_rasters = curate_data.extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, params)
    else:
        labelled_fixation_rasters = load_data.load_labelled_fixation_rasters(params)

    if params.get('replot_face/eye_vs_obj_violins'):
        response_comp.compute_pre_and_post_fixation_response_to_roi_for_each_unit(
            labelled_fixation_rasters, params)


























