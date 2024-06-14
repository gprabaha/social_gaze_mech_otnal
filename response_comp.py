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

import util  # Import the date function
import plotter


def calculate_roi_response_for_unit(unit, filtered_data, output_base_dir):
    unit_data = filtered_data[filtered_data['uuid'] == unit]
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
        pre_spikes = np.array([raster[:500] for raster in roi_data['raster']])
        post_spikes = np.array([raster[500:1000] for raster in roi_data['raster']])
        pre_mean = pre_spikes.mean(axis=1).mean()
        post_mean = post_spikes.mean(axis=1).mean()
        pre_error = pre_spikes.mean(axis=1).std() / np.sqrt(len(pre_spikes))
        post_error = post_spikes.mean(axis=1).std() / np.sqrt(len(post_spikes))
        pre_means.append(pre_mean)
        post_means.append(post_mean)
        pre_errors.append(pre_error)
        post_errors.append(post_error)
    # Perform statistical comparisons
    significant_pre = np.zeros((len(rois), len(rois)), dtype=bool)
    significant_post = np.zeros((len(rois), len(rois)), dtype=bool)

    for i, roi1 in enumerate(rois):
        for j, roi2 in enumerate(rois):
            if i >= j:
                continue

            roi1_pre = np.array([raster[:500] for raster in unit_data[unit_data['fix_roi'] == roi1]['raster']]).mean(axis=1)
            roi2_pre = np.array([raster[:500] for raster in unit_data[unit_data['fix_roi'] == roi2]['raster']]).mean(axis=1)
            t_stat_pre, p_val_pre = ttest_ind(roi1_pre, roi2_pre)
            if p_val_pre < 0.05:
                significant_pre[i, j] = True

            roi1_post = np.array([raster[500:1000] for raster in unit_data[unit_data['fix_roi'] == roi1]['raster']]).mean(axis=1)
            roi2_post = np.array([raster[500:1000] for raster in unit_data[unit_data['fix_roi'] == roi2]['raster']]).mean(axis=1)
            t_stat_post, p_val_post = ttest_ind(roi1_post, roi2_post)
            if p_val_post < 0.05:
                significant_post[i, j] = True

    # Call the plotting function from the other script
    plotter.plot_unit_response_to_rois(unit, rois, pre_means, post_means, pre_errors, post_errors, significant_pre, significant_post, output_dir)

def compute_pre_and_post_fixation_response_to_roi_for_each_unit(labelled_fixation_rasters, params):
    root_dir = params['root_data_dir']
    use_parallel = params.get('use_parallel', False)
    output_base_dir = util.add_date_dir_to_path(os.path.join(root_dir, 'plots', 'roi_fr_response'))

    # Filter the data for 'mon_down' blocks and rasters aligned to 'start_time'
    filtered_data = labelled_fixation_rasters[
        (labelled_fixation_rasters['block'] == 'mon_down') &
        (labelled_fixation_rasters['aligned_to'] == 'start_time')]
    unique_units = filtered_data['uuid'].unique()

    if use_parallel:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(calculate_roi_response_for_unit, unit, filtered_data, output_base_dir): unit for unit in unique_units}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ROI response computed for unit"):
                future.result()
    else:
        for unit in tqdm(unique_units, desc="ROI response computed for unit"):
            calculate_roi_response_for_unit(unit, filtered_data, output_base_dir)



