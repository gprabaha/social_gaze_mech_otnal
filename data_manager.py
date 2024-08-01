#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:42:48 2024

@author: pg496
"""

import logging
import os
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import numexpr as ne

import curate_data
import util
import load_data
import response_comp
import plotter
import fix_and_saccades

import pdb


"""
Big changes needed in session plotter. currently it is appending each row as a separate event which is true
but we have to collect each event in a run and then plot them out. This is too complicated for chatgpt 
so write manually
"""

class DataManager:
    def __init__(self, params):
        self.params = params
        self.setup_logger()
        self.initialize_variables()
        self.find_n_cores()


    def setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)


    def initialize_variables(self):
        """Initialize all necessary variables for the DataManager."""
        self.labelled_gaze_positions_m1 = None
        self.gaze_positions_m1 = None
        self.gaze_position_labels_m1 = None
        self.toy_data = None
        self.labelled_fixations_m1 = None
        self.labelled_saccades_m1 = None
        self.combined_behav_m1 = None
        self.events_within_attention_frame_m1 = None
        self.labelled_spiketimes = None
        self.labelled_fixation_rasters = None


    def find_n_cores(self):
        """Determine the number of CPU cores available, prioritizing SLURM if available."""
        try:
            slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
            num_cpus = int(slurm_cpus)
            print(f"SLURM detected {num_cpus} CPUs")
        except Exception as e:
            print(f"Failed to detect cores with SLURM_CPUS_ON_NODE: {e}")
            num_cpus = None
        if num_cpus is None or num_cpus <= 1:
            num_cpus = multiprocessing.cpu_count()
            print(f"multiprocessing detected {num_cpus} CPUs")
        os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus)
        self.num_cpus = num_cpus
        self.params['num_cpus'] = num_cpus
        print(f"NumExpr set to use {num_cpus} threads")


    def get_or_load_variable(self, variable_name, load_function, compute_function):
        """Load or compute variables based on the given flags in params.
        Args:
            variable_name (str): The name(s) of the variable(s) to load or compute.
            load_function (callable): The function to load the variable(s).
            compute_function (callable): The function to compute the variable(s).
        Returns:
            The loaded or computed variable(s).
        """
        variable_names = [name.strip() for name in variable_name.split(',')]
        should_recompute_flags = [self.params.get(
            f'remake_{name}', False) for name in variable_names]
        needs_loading = any(
            should_recompute_flags) or any(
            getattr(self, name) is None for name in variable_names)
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


    def split_gaze_data(self):
        """Split labelled gaze positions into separate gaze positions and labels."""
        self.gaze_positions_m1 = [item[0] for item in self.labelled_gaze_positions_m1]
        self.gaze_position_labels_m1 = [item[1] for item in self.labelled_gaze_positions_m1]


    def add_frame_of_attention_and_plotting_frame_to_gaze_labels(self):
        """Add frame of attention and plotting frame to gaze position labels."""
        for session_data in self.gaze_position_labels_m1:
            bboxes = session_data['roi_bb_corners']
            frame = util.define_frame_of_attention(bboxes)
            session_data['frame_of_attention'] = frame
            plotting_frame = util.remap_source_coords(
                frame, self.params, 'stretch_from_center_of_mass', 1.3)
            session_data['plotting_frame'] = plotting_frame


    def plot_all_behavior_in_all_sessions(self, use_parallel=False):
        """Plot all behavior within the frame of attention for all sessions.
        Utilizes parallel processing if specified.
        Args:
            use_parallel (bool): Flag to determine if parallel processing should be used.
        """
        root_data_dir = self.params['root_data_dir']
        plots_dir = util.add_date_dir_to_path(
            os.path.join(root_data_dir, 'plots', 'fix_and_saccades_all_sessions'))
        os.makedirs(plots_dir, exist_ok=True)
        sessions = list(self.events_within_attention_frame_m1['session_name'].unique())


        # !!!!!!!!!!!!!!!!!!
        sessions = sessions[1:]



        if use_parallel:
            with Pool() as pool:
                for _ in tqdm(pool.starmap(
                        plotter.plot_behavior_for_session,
                        [(session, self.events_within_attention_frame_m1,
                          self.gaze_position_labels_m1, plots_dir) for session in sessions]
                    ), total=len(sessions),
                    desc='Plotting behavior for session in parallel'):
                    pass
                pool.close()
                pool.join()
        else:
            for session in tqdm(sessions, total=len(sessions),
                                desc='Plotting behavior for session in serial'):
                plotter.plot_behavior_for_session(
                    session, self.events_within_attention_frame_m1,
                    self.gaze_position_labels_m1, plots_dir)


    def run(self):
        """Run the DataManager to fetch, process, and plot gaze and behavioral data."""
        _, self.params = util.fetch_root_data_dir(self.params)
        _, self.params = util.fetch_data_source_dir(self.params)
        _, self.params = util.fetch_session_subfolder_paths_from_source(self.params)
        _, self.params = util.fetch_processed_data_dir(self.params)
        self.params['num_cpus'] = self.num_cpus
        self.labelled_gaze_positions_m1 = self.get_or_load_variable(
            'labelled_gaze_positions_m1',
            load_data.load_labelled_gaze_positions,
            lambda p: curate_data.extract_labelled_gaze_positions_m1(p))
        self.logger.info(f"M1 remapped gaze pos data acquired!")
        self.split_gaze_data()
        self.logger.info(f"Gaze data split into: self.gaze_positions and self.gaze_position_labels!")
        self.add_frame_of_attention_and_plotting_frame_to_gaze_labels()
        self.logger.info(f"Frame of attention and plotting added to gaze data")
        if self.params['use_toy_data']:
            self.logger.info(f"!! USING TOY DATA !!")
            self.toy_data = self.get_or_load_variable(
                'toy_data',
                load_data.load_toy_data,
                lambda p: curate_data.generate_toy_gazepos_data(self.labelled_gaze_positions_m1, p))
            input_data = self.toy_data
        else:
            input_data = self.labelled_gaze_positions_m1
        self.labelled_fixations_m1, self.labelled_saccades_m1, self.combined_behav_m1 = self.get_or_load_variable(
            'labelled_fixations_m1, labelled_saccades_m1, combined_behav_m1',
            load_data.load_m1_labelled_fixations_saccades_and_combined,
            lambda p: curate_data.extract_fixations_and_saccades_with_labels(input_data, p))
        self.logger.info(f"M1 fixations and saccades acquired")
        self.events_within_attention_frame_m1 = curate_data.isolate_events_within_attention_frame(
            self.combined_behav_m1, self.labelled_gaze_positions_m1, use_parallel=True)
        self.events_within_attention_frame_m1.head()
        self.logger.info(f"Events within attention frame isolated")
        if self.params['make_plots']:
            self.plot_all_behavior_in_all_sessions(use_parallel=False)
            self.logger.info(f"Plots generated successfully")

        # plotter.plot_fixation_proportions_for_diff_conditions(self.labelled_fixations_m1, self.params)
        # plotter.plot_fixation_heatmaps(self.labelled_fixations_m1, self.params)
           
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