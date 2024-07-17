#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""


import logging
import numpy as np

import curate_data
import util
import load_data
import response_comp

import pdb

class DataManager:
    def __init__(self, params):
        self.params = params
        self.setup_logger()
        self.initialize_variables()

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


    def initialize_variables(self):
        self.labelled_gaze_positions_m1 = None
        self.toy_data = None
        self.labelled_fixations_m1 = None
        self.labelled_saccades_m1 = None
        self.labelled_spiketimes = None
        self.labelled_fixation_rasters = None


    def get_or_load_variable(self, variable_name, load_function, compute_function):
        variable_names = [name.strip() for name in variable_name.split(',')]
        should_recompute_flags = [self.params.get(f'remake_{name}', False) for name in variable_names]
        needs_loading = any(should_recompute_flags) or any(getattr(self, name) is None for name in variable_names)

        if needs_loading:
            if any(should_recompute_flags):
                self.logger.info(f"Recomputing variable(s): {variable_name}")
                result = compute_function(self.params)
            else:
                self.logger.info(f"Loading variable(s): {variable_name}")
                result = load_function(self.params)
            
            if isinstance(result, tuple):
                if len(result) != len(variable_names):
                    raise ValueError(f"Expected {len(variable_names)} values, but got {len(result)}")
                for name, value in zip(variable_names, result):
                    setattr(self, name, value)
            else:
                setattr(self, variable_names[0], result)

        if len(variable_names) == 1:
            return getattr(self, variable_names[0])
        else:
            return tuple(getattr(self, name) for name in variable_names)


    def human_readable_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024


    def run(self):
        root_data_dir, self.params = util.fetch_root_data_dir(self.params)
        data_source_dir, self.params = util.fetch_data_source_dir(self.params)
        session_paths, self.params = util.fetch_session_subfolder_paths_from_source(self.params)
        processed_data_dir, self.params = util.fetch_processed_data_dir(self.params)

        self.labelled_gaze_positions_m1 = self.get_or_load_variable(
            'labelled_gaze_positions_m1',
            load_data.load_labelled_gaze_positions,
            lambda p: curate_data.extract_labelled_gaze_positions_m1(p)
        )
        self.logger.info(f"M1 remapped gaze pos data acquired!")

        if self.params['use_toy_data']:
            self.logger.info(f"!! USING TOY DATA !!")
            self.toy_data = self.get_or_load_variable(
                'toy_data',
                load_data.load_toy_data,
                lambda p: curate_data.generate_toy_gazepos_data(self.labelled_gaze_positions_m1, p)
            )
            input_data = self.toy_data
        else:
            input_data = self.labelled_gaze_positions_m1

        self.labelled_fixations_m1, self.labelled_saccades_m1 = self.get_or_load_variable(
            'labelled_fixations_m1, labelled_saccades_m1',
            load_data.load_m1_labelled_fixations_and_saccades,
            lambda p: curate_data.extract_fixations_and_saccades_with_labels(input_data, p)
        )
        self.logger.info(f"M1 fixations and saccades acquired")
        pdb.set_trace()
        
        # self.labelled_saccades_m1 = self.get_or_load_variable(
        #     'labelled_saccades_m1',
        #     load_data.load_saccade_labels,
        #     lambda p: curate_data.extract_saccades_with_labels(self.labelled_gaze_positions_m1, p)
        # )
        
        # self.labelled_spiketimes = self.get_or_load_variable(
        #     'labelled_spiketimes',
        #     load_data.load_processed_spiketimes,
        #     lambda p: curate_data.extract_spiketimes_for_all_sessions(p)
        # )
        
        # self.labelled_fixation_rasters = self.get_or_load_variable(
        #     'labelled_fixation_rasters',
        #     load_data.load_labelled_fixation_rasters,
        #     lambda p: curate_data.extract_fixation_raster(session_paths, self.labelled_fixations, self.labelled_spiketimes, p)
        # )


        # if self.params.get('replot_face/eye_vs_obj_violins'):
        #     response_comp.compute_pre_and_post_fixation_response_to_roi_for_each_unit(self.labelled_fixation_rasters, self.params)


def main():
    params = util.get_params()
    params.update({
        'parallelize_local_reclustering_over_n_fixations': False,
        'submit_separate_jobs_for_sessions': True,
        'use_toy_data': False,
        'remake_toy_data': False,
        'is_cluster': True,
        'use_parallel': True,
        'remake_labelled_gaze_positions_m1': False,
        'fixation_detection_method': 'cluster_fix',
        'remake_labelled_fixations_m1': True,
        'remake_labelled_saccades_m1': True,
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
        'reload_existing_unit_roi_comp_stats': False
    })

    data_manager = DataManager(params)
    data_manager.run()

if __name__ == "__main__":
    main()





















