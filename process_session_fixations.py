#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:37:42 2024

@author: pg496
"""

import argparse
import logging
import pickle
import os
import pandas as pd
from fix import extract_or_load_fixations_and_saccades
import load_data
import util

def main(session_index):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load parameters
    params = util.get_params()
    params.update({
        'is_cluster': True,
        'use_parallel': False,
        'remake_labelled_gaze_positions_m1': False,
        'fixation_detection_method': 'cluster_fix',
        'remake_labelled_fixations': True,
        'remake_labelled_saccades_m1': False,
        'remake_labelled_spiketimes': False,
        'remake_labelled_fixation_rasters': True,
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
        'use_existing_variables': False,
        'reload_existing_unit_roi_comp_stats': False,
        'submit_separate_jobs_for_session_raster': True
    })

    logging.info(f"Starting fixation detection for session index: {session_index}")

    # Load labelled gaze positions
    labelled_gaze_positions = load_data.load_labelled_gaze_positions(params)

    session_data = labelled_gaze_positions[session_index]

    # Extract fixations and saccades
    all_fix_timepos, fix_detection_results, saccade_detection_results = extract_or_load_fixations_and_saccades([session_data], params)
    
    # Save results
    output_dir = params['processed_data_dir']
    fixations_file = os.path.join(output_dir, f"{session_index}_fixations.pkl")
    
    with open(fixations_file, 'wb') as f:
        pickle.dump((all_fix_timepos, fix_detection_results, saccade_detection_results), f)
    
    logging.info(f"Fixation detection completed for session index: {session_index}")
    logging.info(f"Results saved to: {fixations_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process session fixation detection")
    parser.add_argument('--session_index', type=int, required=True, help='Index of the session in labelled gaze positions list')
    
    args = parser.parse_args()
    main(args.session_index)
