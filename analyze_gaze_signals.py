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

class DataManager:
    def __init__(self, params):
        self.params = params
        self.local_vars = {}
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_or_load_variable(self, variable_name, load_function, compute_function):
        if self.params.get(f'remake_{variable_name}', False) or self.local_vars.get(variable_name) is None:
            if self.params.get(f'remake_{variable_name}', False):
                self.logger.info(f"Recomputing variable: {variable_name}")
                self.local_vars[variable_name] = compute_function(self.params)
            else:
                self.logger.info(f"Loading variable: {variable_name}")
                self.local_vars[variable_name] = load_function(self.params)
        return self.local_vars[variable_name]

    def run(self):
        root_data_dir, self.params = util.fetch_root_data_dir(self.params)
        data_source_dir, self.params = util.fetch_data_source_dir(self.params)
        session_paths, self.params = util.fetch_session_subfolder_paths_from_source(self.params)
        processed_data_dir, self.params = util.fetch_processed_data_dir(self.params)

        labelled_gaze_positions_m1 = self.get_or_load_variable(
            'labelled_gaze_positions_m1',
            load_data.load_labelled_gaze_positions,
            lambda p: curate_data.extract_labelled_gaze_positions_m1(curate_data.get_unique_doses(curate_data.extract_and_update_meta_info(p)))
        )

        labelled_fixations = self.get_or_load_variable(
            'labelled_fixations',
            load_data.load_m1_fixation_labels,
            lambda p: curate_data.extract_fixations_with_labels_parallel(labelled_gaze_positions_m1, p)
        )

        labelled_saccades_m1 = self.get_or_load_variable(
            'labelled_saccades_m1',
            load_data.load_saccade_labels,
            lambda p: curate_data.extract_saccades_with_labels(labelled_gaze_positions_m1, p)
        )

        labelled_spiketimes = self.get_or_load_variable(
            'labelled_spiketimes',
            load_data.load_processed_spiketimes,
            curate_data.extract_spiketimes_for_all_sessions
        )

        labelled_fixation_rasters = self.get_or_load_variable(
            'labelled_fixation_rasters',
            load_data.load_labelled_fixation_rasters,
            lambda p: curate_data.extract_fixation_raster(session_paths, labelled_fixations, labelled_spiketimes, p)
        )

        if self.params.get('replot_face/eye_vs_obj_violins'):
            response_comp.compute_pre_and_post_fixation_response_to_roi_for_each_unit(labelled_fixation_rasters, self.params)

def main():
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

























