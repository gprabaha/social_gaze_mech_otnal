#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:13:07 2024

@author: pg496
"""

import os
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


import matplotlib.pyplot as plt
import seaborn as sns


import util  # Import the date function
import plotter


from util import add_date_dir_to_path
from plotter import plot_unit_response_to_rois

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_roi_response_for_unit(unit, filtered_data, output_base_dir):
    try:
        unit_data = filtered_data[filtered_data['uuid'] == unit]
        if unit_data.empty:
            logger.info(f"No data for unit {unit}, skipping.")
            return
        rois = unit_data['fix_roi'].unique()
        region = unit_data['region'].iloc[0]  # Assuming region is consistent for a unit
        output_dir = os.path.join(output_base_dir, region)
        os.makedirs(output_dir, exist_ok=True)
        pre_means = []
        post_means = []
        pre_errors = []
        post_errors = []
        for roi in rois:
            roi_data = unit_data[unit_data['fix_roi'] == roi]
            if roi_data.empty:
                logger.info(f"No data for unit {unit} in ROI {roi}, skipping ROI.")
                continue
            pre_spikes = np.array([raster[:500] for raster in roi_data['raster']])
            post_spikes = np.array([raster[500:1000] for raster in roi_data['raster']])
            if pre_spikes.ndim != 2 or post_spikes.ndim != 2:
                logger.warning(f"Unexpected array dimensions for unit {unit}, ROI {roi}: pre_spikes {pre_spikes.shape}, post_spikes {post_spikes.shape}. Skipping ROI.")
                continue
            if pre_spikes.size == 0 or post_spikes.size == 0:
                logger.info(f"No pre or post spikes for unit {unit} in ROI {roi}, skipping ROI.")
                continue
            pre_mean = pre_spikes.mean(axis=1).mean()
            post_mean = post_spikes.mean(axis=1).mean()
            pre_error = pre_spikes.mean(axis=1).std() / np.sqrt(len(pre_spikes))
            post_error = post_spikes.mean(axis=1).std() / np.sqrt(len(post_spikes))
            pre_means.append(pre_mean)
            post_means.append(post_mean)
            pre_errors.append(pre_error)
            post_errors.append(post_error)
        if not pre_means or not post_means:
            logger.info(f"No valid data to plot for unit {unit}, skipping.")
            return
        # Perform statistical comparisons
        significant_pre = np.zeros((len(rois), len(rois)), dtype=bool)
        significant_post = np.zeros((len(rois), len(rois)), dtype=bool)
        for i, roi1 in enumerate(rois):
            for j, roi2 in enumerate(rois):
                if i >= j:
                    continue
                roi1_pre = np.array([raster[:500] for raster in unit_data[unit_data['fix_roi'] == roi1]['raster']]).mean(axis=1)
                roi2_pre = np.array([raster[:500] for raster in unit_data[unit_data['fix_roi'] == roi2]['raster']]).mean(axis=1)
                if roi1_pre.ndim != 1 or roi2_pre.ndim != 1:
                    logger.warning(f"Unexpected array dimensions for statistical comparison: roi1_pre {roi1_pre.shape}, roi2_pre {roi2_pre.shape}. Skipping comparison.")
                    continue
                t_stat_pre, p_val_pre = ttest_ind(roi1_pre, roi2_pre)
                if p_val_pre < 0.05:
                    significant_pre[i, j] = True
                roi1_post = np.array([raster[500:1000] for raster in unit_data[unit_data['fix_roi'] == roi1]['raster']]).mean(axis=1)
                roi2_post = np.array([raster[500:1000] for raster in unit_data[unit_data['fix_roi'] == roi2]['raster']]).mean(axis=1)
                if roi1_post.ndim != 1 or roi2_post.ndim != 1:
                    logger.warning(f"Unexpected array dimensions for statistical comparison: roi1_post {roi1_post.shape}, roi2_post {roi2_post.shape}. Skipping comparison.")
                    continue
                t_stat_post, p_val_post = ttest_ind(roi1_post, roi2_post)
                if p_val_post < 0.05:
                    significant_post[i, j] = True
        # Call the plotting function from the other script
        plot_unit_response_to_rois(unit, rois, pre_means, post_means, pre_errors, post_errors, significant_pre, significant_post, output_dir)
    except Exception as e:
        logger.error(f"Error processing unit {unit}: {e}")


def compute_pre_and_post_fixation_response_to_roi_for_each_unit(labelled_fixation_rasters, params):
    root_dir = params['root_data_dir']
    use_parallel = params.get('use_parallel', False)
    output_base_dir = add_date_dir_to_path(os.path.join(root_dir, 'plots', 'roi_fr_response'))
    # Filter the data for 'mon_down' blocks and rasters aligned to 'start_time'
    filtered_data = labelled_fixation_rasters[
        (labelled_fixation_rasters['block'] == 'mon_down') &
        (labelled_fixation_rasters['aligned_to'] == 'start_time')]
    unique_units = filtered_data['uuid'].unique()
    if use_parallel:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(calculate_roi_response_for_unit, unit, filtered_data, output_base_dir): unit for unit in unique_units}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ROI response computed for unit"):
                future.result()
    else:
        for unit in tqdm(unique_units, desc="ROI response computed for unit"):
            calculate_roi_response_for_unit(unit, filtered_data, output_base_dir)


def compare_roi_responses_for_all_units(labelled_fixation_rasters, params):
    root_dir = params['root_data_dir']
    use_parallel = params.get('use_parallel', False)
    output_base_dir = add_date_dir_to_path(os.path.join(root_dir, 'plots', 'roi_response_comparison_each_unit'))
    
    # Filter the data for 'mon_down' blocks and rasters aligned to 'start_time'
    filtered_data = labelled_fixation_rasters[
        (labelled_fixation_rasters['block'] == 'mon_down') &
        (labelled_fixation_rasters['aligned_to'] == 'start_time')]
    
    unique_units = filtered_data['uuid'].unique()
    if use_parallel:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(compare_roi_responses_for_unit, unit, filtered_data, output_base_dir): unit for unit in unique_units}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ROI response comparison computed for unit"):
                future.result()
    else:
        for unit in tqdm(unique_units, desc="ROI response comparison computed for unit"):
            compare_roi_responses_for_unit(unit, filtered_data, output_base_dir)


def compare_roi_responses_for_unit(unit, filtered_data, output_base_dir):
    try:
        unit_data = filtered_data[filtered_data['uuid'] == unit]
        if unit_data.empty:
            logger.info(f"No data for unit {unit}, skipping.")
            return
        rois = ['eye_bbox', 'left_obj_bbox', 'right_obj_bbox', 'face_bbox']
        region = unit_data['region'].iloc[0]  # Assuming region is consistent for a unit
        output_dir = os.path.join(output_base_dir, region)
        os.makedirs(output_dir, exist_ok=True)
        pre_data = {}
        post_data = {}
        for roi in rois:
            roi_data = unit_data[unit_data['fix_roi'] == roi]
            if roi_data.empty:
                logger.info(f"No data for unit {unit} in ROI {roi}, skipping ROI.")
                continue
            pre_data[roi] = np.array([raster[:500] for raster in roi_data['raster']])
            post_data[roi] = np.array([raster[500:1000] for raster in roi_data['raster']])
        if not pre_data or not post_data:
            logger.info(f"No valid data to plot for unit {unit}, skipping.")
            return
        plotter.plot_roi_comparisons_for_unit(unit, region, pre_data, post_data, output_dir)
    except Exception as e:
        logger.error(f"Error processing unit {unit}: {e}")











