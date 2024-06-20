#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:13:07 2024

@author: pg496
"""


import os
import numpy as np
from scipy.stats import ttest_ind
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from collections import defaultdict
import pickle


from tqdm import tqdm


import util  # Import the date function
import plotter


import pdb

# Configure logging
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


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
        plotter.plot_unit_response_to_rois(unit, rois, pre_means, post_means, pre_errors, post_errors, significant_pre, significant_post, output_dir)
    except Exception as e:
        logger.error(f"Error processing unit {unit}: {e}")





# Assuming the existence of util and plotter modules and a logger

def default_dict_function():
    return {
        'pre': defaultdict(list), 
        'post': defaultdict(list), 
        'both': defaultdict(list), 
        'neither': defaultdict(list),
        'either': defaultdict(list)
    }

def merge_results(main_results, new_results):
    for region, region_data in new_results.items():
        for key in region_data:
            for comp, units in region_data[key].items():
                main_results[region][key][comp].extend(units)

def compute_pre_and_post_fixation_response_to_roi_for_each_unit(labelled_fixation_rasters, params):
    root_dir = params['root_data_dir']
    use_parallel = params.get('use_parallel', False)
    output_base_dir = util.add_date_dir_to_path(os.path.join(
        root_dir, 'plots', 'roi_response_comparison_each_unit'))
    processed_data_file = os.path.join(root_dir, 'processed_data', 'roi_spike_count_comparison_for_each_unit.pkl')

    # Filter the data for 'mon_down' blocks and rasters aligned to 'start_time'
    filtered_data = labelled_fixation_rasters[
        (labelled_fixation_rasters['block'] == 'mon_down') &
        (labelled_fixation_rasters['aligned_to'] == 'start_time')]
    unique_units = filtered_data['uuid'].unique()
    results = defaultdict(default_dict_function)

    if params.get('reload_existing_unit_roi_comp_stats') and os.path.exists(processed_data_file):
        with open(processed_data_file, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"Loaded existing results from {processed_data_file}")
    else:
        if use_parallel:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(analyze_and_plot_unit, unit, filtered_data, output_base_dir): unit for unit in unique_units}
                for future in tqdm(as_completed(futures), total=len(futures), desc="ROI response comparison computed for unit"):
                    unit_results = future.result()
                    merge_results(results, unit_results)
        else:
            for unit in tqdm(unique_units, desc="ROI response comparison computed for unit"):
                unit_results = analyze_and_plot_unit(unit, filtered_data, output_base_dir)
                merge_results(results, unit_results)
        
        with open(processed_data_file, 'wb') as f:
            pickle.dump(results, f)
            logger.info(f"Saved results to {processed_data_file}")

    # Generate summary plots for each region
    summary_output_dir = output_base_dir  # Date-labelled directory for summary plots
    for region in results.keys():
        plotter.plot_pie_charts(region, results[region], summary_output_dir)
        plotter.plot_venn_diagrams(region, results[region], summary_output_dir)

def analyze_and_plot_unit(unit, filtered_data, output_base_dir):
    unit_results = defaultdict(default_dict_function)
    try:
        unit_data = filtered_data[filtered_data['uuid'] == unit]
        if unit_data.empty:
            logger.info(f"No data for unit {unit}, skipping.")
            return unit_results
        rois = ['eye_bbox', 'left_obj_bbox', 'right_obj_bbox', 'face_bbox']
        region = unit_data['region'].iloc[0]  # Assuming region is consistent for a unit
        unit_output_dir = os.path.join(output_base_dir, region)
        os.makedirs(unit_output_dir, exist_ok=True)
        pre_data = {}
        post_data = {}
        for roi in rois:
            roi_data = unit_data[unit_data['fix_roi'] == roi]
            if roi_data.empty:
                logger.info(f"No data for unit {unit} in ROI {roi}, skipping ROI.")
                continue
            pre_data[roi] = np.array([raster[:500] for raster in roi_data['raster']])
            post_data[roi] = np.array([raster[500:] for raster in roi_data['raster']])
        if not pre_data or not post_data:
            logger.info(f"No valid data to plot for unit {unit}, skipping.")
            return unit_results
        significant_pre, significant_post, significant_both, significant_neither, significant_either = \
            analyze_significant_differences(unit, region, pre_data, post_data, output_base_dir)
        # Update unit_results
        for comp in significant_pre.keys():
            unit_results[region]['pre'][comp].extend(significant_pre[comp])
            unit_results[region]['post'][comp].extend(significant_post[comp])
            unit_results[region]['both'][comp].extend(significant_both[comp])
            unit_results[region]['neither'][comp].extend(significant_neither[comp])
            unit_results[region]['either'][comp].extend(significant_either[comp])
        plotter.plot_roi_comparisons_for_unit(unit, region, pre_data, post_data, unit_output_dir)
    except Exception as e:
        logger.error(f"Error processing unit {unit}: {e}")
    return unit_results

def analyze_significant_differences(unit, region, pre_data, post_data, output_dir):
    comparisons = [
        ('eye_bbox', 'left_obj_bbox'),
        ('eye_bbox', 'right_obj_bbox'),
        ('eye_bbox', 'left_right_combined'),
        ('face_bbox', 'left_obj_bbox'),
        ('face_bbox', 'right_obj_bbox'),
        ('face_bbox', 'left_right_combined')
    ]
    significant_pre = defaultdict(list)
    significant_post = defaultdict(list)
    significant_both = defaultdict(list)
    significant_neither = defaultdict(list)
    significant_either = defaultdict(list)

    for roi1, roi2 in comparisons:
        if roi2 == 'left_right_combined':
            pre_data_combined = np.concatenate((pre_data['left_obj_bbox'], pre_data['right_obj_bbox']), axis=0)
            post_data_combined = np.concatenate((post_data['left_obj_bbox'], post_data['right_obj_bbox']), axis=0)
        else:
            pre_data_combined = pre_data[roi2]
            post_data_combined = post_data[roi2]
        
        if pre_data[roi1].size == 0 or pre_data_combined.size == 0 or post_data[roi1].size == 0 or post_data_combined.size == 0:
            continue  # Skip if any of the data arrays are empty
        
        data = [
            pre_data[roi1].mean(axis=1).astype(float), pre_data_combined.mean(axis=1).astype(float),
            post_data[roi1].mean(axis=1).astype(float), post_data_combined.mean(axis=1).astype(float)
        ]
        
        if np.isnan(data[0]).all() or np.isnan(data[1]).all() or np.isnan(data[2]).all() or np.isnan(data[3]).all():
            continue  # Skip if any of the data arrays contain only NaNs
        
        _, p_val_pre = ttest_ind(data[0], data[1], nan_policy='omit')
        _, p_val_post = ttest_ind(data[2], data[3], nan_policy='omit')

        comparison_key = roi1 + " vs " + roi2

        if p_val_pre < 0.05 and p_val_post < 0.05:
            significant_both[comparison_key].append(unit)
        elif p_val_pre < 0.05 and p_val_post >= 0.05:
            significant_pre[comparison_key].append(unit)
        elif p_val_pre >= 0.05 and p_val_post < 0.05:
            significant_post[comparison_key].append(unit)
        else:
            significant_neither[comparison_key].append(unit)
        
        if p_val_pre < 0.05 or p_val_post < 0.05:
            significant_either[comparison_key].append(unit)
    
    results = {
        'pre': significant_pre,
        'post': significant_post,
        'both': significant_both,
        'neither': significant_neither,
        'either': significant_either
    }

    # Save the results dictionary
    with open(os.path.join(output_dir, f'{region}_significant_units.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return significant_pre, significant_post, significant_both, significant_neither, significant_either



















