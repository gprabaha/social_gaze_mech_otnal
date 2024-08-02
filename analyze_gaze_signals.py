#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""

import logging
import util
from data_manager import DataManager


'''
Numpy arrays being exported as a single cell in a dataframe are often being stored as strings
After retreiving them we are currently doing operations to convert the recovered string to the
expected array using util.convert_to_array. But this is very unstable. Ensure that all
single coordinate or 2D array containing sets of coordinates are exported and imported in a
uniform manner
'''

def main():
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    params = util.get_params()
    params.update({
        'num_cpus': None,
        'parallelize_local_reclustering_over_n_fixations': False,
        'do_local_reclustering_in_parallel': False,
        'submit_separate_jobs_for_sessions': True,
        'use_toy_data': False,
        'remake_toy_data': False,
        'is_cluster': True,
        'use_parallel': True,
        'remake_labelled_gaze_positions_m1': False,
        'fixation_detection_method': 'cluster_fix',
        'remake_labelled_fixations_m1': False,
        'remake_labelled_saccades_m1': False,
        'remake_combined_behav_m1': False,
        'remake_labelled_spiketimes': False,
        'remake_labelled_fixation_rasters': True,
        'make_plots': True,
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
        'reload_existing_unit_roi_comp_stats': False
    })
    data_manager = DataManager(params)
    data_manager.run()


if __name__ == "__main__":
    main()





















