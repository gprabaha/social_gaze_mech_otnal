#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:35:03 2024

@author: pg496
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import os
from scipy.stats import ttest_ind
import seaborn as sns
from datetime import datetime

import util
import load_data

import pdb


def plot_fixation_proportions_for_diff_conditions(params):
    """
    Plots the proportion of fixations on different ROIs for different conditions.

    Parameters:
    - params (dict): Dictionary containing parameters.
    """
    root_data_dir = params['root_data_dir']
    if params.get('export_plots_to_local_folder', True):
        plots_dir = 'plots'
    else:
        plots_dir = os.path.join(root_data_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    remap_flag = util.get_filename_flag_info(params)
    fixation_labels_m1 = load_data.load_m1_fixation_labels(params)
    # Filtering out discarded runs
    valid_runs = fixation_labels_m1[fixation_labels_m1['block']
                                    != 'discard']
    conditions = {
        'mon_up': valid_runs['block'] == 'mon_up',
        'mon_down': valid_runs['block'] == 'mon_down'}
    agents = {
        'Lynch': valid_runs['agent'] == 'Lynch',
        'Tarantino': valid_runs['agent'] == 'Tarantino'}
    rois = ['face_bbox', 'eye_bbox', 'left_obj_bbox', 'right_obj_bbox']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(
        f'Proportion of Fixations on Different ROIs {remap_flag}',
        fontsize=16)
    for i, (agent_name, agent_cond) in enumerate(agents.items()):
        for j, (block_name, block_cond) in enumerate(conditions.items()):
            ax = axes[i, j]
            data = valid_runs[agent_cond & block_cond]
            proportions = [np.mean(data['fix_roi'] == roi)
                           for roi in rois]
            ax.bar(rois, proportions, color=[
                'blue', 'orange', 'green', 'red'])
            ax.set_title(f'{agent_name} - {block_name}')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Proportion of Fixations')
            ax.set_xlabel('ROI')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = f'fixation_proportions{remap_flag}.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()


def plot_gaze_heatmaps(params):
    """
    Generates and saves gaze heatmaps for different conditions.
    Parameters:
    - params (dict): Dictionary containing parameters.
    """
    root_data_dir = params['root_data_dir']
    if params.get('export_plots_to_local_folder', True):
        plots_dir = util.add_date_dir_to_path('plots/gaze_heatmaps')
    else:
        plots_dir = util.add_date_dir_to_path(
            os.path.join(root_data_dir, 'plots', 'gaze_heatmaps'))
    os.makedirs(plots_dir, exist_ok=True)
    labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(
        params)
    plot_gaze_heatmaps_for_all_sessions(
        labelled_gaze_positions_m1, params, plots_dir)


def plot_gaze_heatmaps_for_all_sessions(labelled_gaze_positions_m1,
                                        params, plots_dir):
    for session_idx, (gaze_positions, session_info) \
        in enumerate(tqdm(labelled_gaze_positions_m1,
                          desc="Processing Sessions")):
        plot_gaze_heatmap_for_one_session(
            gaze_positions, session_info, session_idx, plots_dir)


def plot_gaze_heatmap_for_one_session(gaze_positions, session_info,
                                      session_idx, plots_dir):
    sampling_rate = session_info['sampling_rate']
    start_times = session_info['startS']
    stop_times = session_info['stopS']
    roi_bb_corners = session_info['roi_bb_corners']
    # Combine all runs into one for plotting
    all_gaze_positions = np.vstack(
        [gaze_positions[round(start / sampling_rate):
                        round(stop / sampling_rate)]
         for start, stop in zip(start_times, stop_times)])
    # Generate the 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        all_gaze_positions[:, 0], all_gaze_positions[:, 1], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap.T, extent=extent,
               origin='lower', cmap='hot', aspect='auto')
    # Plot ROI bounding boxes with diff colors
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'gray']
    for i, (roi_name, corners) in enumerate(roi_bb_corners.items()):
        bottomLeft = corners['bottomLeft']
        topRight = corners['topRight']
        width = abs(topRight[0] - bottomLeft[0])
        height = abs(topRight[1] - bottomLeft[1])
        rect = Rectangle((bottomLeft[0], bottomLeft[1]), width, height,
                         fill=False, edgecolor=colors[i % len(colors)],
                         linewidth=2, label=roi_name)
        plt.gca().add_patch(rect)
    plt.colorbar(label='Frequency')
    plt.title(f'Session {session_info["session_name"]}, Number of Runs: {len(start_times)}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(loc='upper right')
    # Save the plot
    plot_filename = f'session_{session_info["session_name"]}.png'
    plt.savefig(os.path.join(plots_dir, plot_filename))
    plt.close()


def plot_fixation_heatmaps(params):
    root_data_dir = params['root_data_dir']
    if params.get('export_plots_to_local_folder', True):
        plots_dir = util.add_date_dir_to_path('plots/fix_heatmaps')
    else:
        plots_dir = util.add_date_dir_to_path(
            os.path.join(root_data_dir, 'plots', 'fix_heatmaps'))
    os.makedirs(plots_dir, exist_ok=True)
    labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(
        params)
    all_fixation_labels = load_data.load_m1_fixation_labels(params)
    plot_fixation_heatmaps_for_all_sessions(
            all_fixation_labels, labelled_gaze_positions_m1, params, plots_dir)


def plot_fixation_heatmaps_for_all_sessions(
        all_fixation_labels, labelled_gaze_positions_m1, params, plots_dir):
    for session_idx, (gaze_positions, session_info) in enumerate(
            tqdm(labelled_gaze_positions_m1, desc="Processing Sessions")):
        session_name = session_info['session_name']
        roi_bb_corners = session_info['roi_bb_corners']
        session_fixations = all_fixation_labels[
            all_fixation_labels['session_name'] == session_name]
        plot_fixation_heatmap_for_one_session(
            session_fixations, roi_bb_corners, session_name, plots_dir)


def plot_fixation_heatmap_for_one_session(session_fixations, roi_bb_corners,
                                          session_name, plots_dir):
    # Generate the 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        session_fixations['mean_x_pos'],
        session_fixations['mean_y_pos'],
        bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap.T, extent=extent, origin='lower',
               cmap='hot', aspect='auto')
    # Plot ROI bounding boxes with different colors
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'gray']
    for i, (roi_name, corners) in enumerate(roi_bb_corners.items()):
        bottom_left = corners['bottomLeft']
        top_right = corners['topRight']
        width = abs(top_right[0] - bottom_left[0])
        height = abs(top_right[1] - bottom_left[1])
        rect = Rectangle((bottom_left[0], bottom_left[1]), width, height,
                         fill=False, edgecolor=colors[i % len(colors)],
                         linewidth=2, label=roi_name)
        plt.gca().add_patch(rect)
    plt.colorbar(label='Frequency')
    plt.title(f'Session {session_name}, Number of Fixations: {len(session_fixations)}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(loc='upper right')
    # Save the plot
    plot_filename = f'session_{session_name}_fixations.png'
    plt.savefig(os.path.join(plots_dir, plot_filename))
    plt.close()


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import ttest_ind
import logging

def plot_roi_response_of_each_unit(labelled_fixation_rasters, params):
    """
    Function to plot the mean ROI response of each unit.
    Parameters:
    labelled_fixation_rasters (pd.DataFrame): DataFrame containing all generated rasters and labels.
    params (dict): Dictionary containing parameters for plotting.
    """
    # Convert necessary columns to appropriate data types
    labelled_fixation_rasters['fix_duration'] = labelled_fixation_rasters['fix_duration'].astype(float)
    labelled_fixation_rasters['mean_x_pos'] = labelled_fixation_rasters['mean_x_pos'].astype(float)
    labelled_fixation_rasters['mean_y_pos'] = labelled_fixation_rasters['mean_y_pos'].astype(float)
    
    # Parameters
    pre_event_time = params.get('raster_pre_event_time', 0.5)
    post_event_time = params.get('raster_post_event_time', 0.5)
    raster_bin_size = params.get('raster_bin_size', 0.01)
    bins_pre = int(pre_event_time / raster_bin_size)
    bins_post = int(post_event_time / raster_bin_size)
    
    # Filter for start_time aligned rasters
    start_time_rasters = labelled_fixation_rasters[labelled_fixation_rasters['aligned_to'] == 'start_time']
    
    # List of ROIs
    rois = start_time_rasters['fix_roi'].unique()
    
    # List of units
    units = start_time_rasters['uuid'].unique()
    
    # Track differentiating neurons for ACC and BLA regions
    acc_diff_neurons = {roi: 0 for roi in rois}
    bla_diff_neurons = {roi: 0 for roi in rois}
    acc_total_neurons = len(start_time_rasters[start_time_rasters['region'] == 'ACC']['uuid'].unique())
    bla_total_neurons = len(start_time_rasters[start_time_rasters['region'] == 'BLA']['uuid'].unique())
    
    # Create directory for plots
    root_data_dir = params['root_data_dir']
    date_label = datetime.now().strftime('%Y-%m-%d')
    plot_dir = os.path.join(root_data_dir, 'plots', 'spike_count_comparison', date_label)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plotting
    for unit in units:
        try:
            unit_data = start_time_rasters[start_time_rasters['uuid'] == unit]
            session_name = unit_data.iloc[0]['session_name']
            region = unit_data.iloc[0]['region']
            fig, axes = plt.subplots(len(rois), 1, figsize=(10, len(rois) * 5))
            fig.suptitle(f'Unit {unit} (Session: {session_name}, Region: {region}) ROI Response')
            for i, roi in enumerate(rois):
                roi_data = unit_data[unit_data['fix_roi'] == roi]
                mon_up = roi_data[roi_data['block'] == 'mon_up']
                mon_down = roi_data[roi_data['block'] == 'mon_down']
                pre_up = np.array([raster[:bins_pre] for raster in mon_up['raster']])
                post_up = np.array([raster[bins_pre:bins_pre + bins_post] for raster in mon_up['raster']])
                pre_down = np.array([raster[:bins_pre] for raster in mon_down['raster']])
                post_down = np.array([raster[bins_pre:bins_pre + bins_post] for raster in mon_down['raster']])
                
                mean_pre_up = np.mean(pre_up, axis=1)
                mean_post_up = np.mean(post_up, axis=1)
                mean_pre_down = np.mean(pre_down, axis=1)
                mean_post_down = np.mean(post_down, axis=1)
                
                mean_mean_pre_up = np.mean(mean_pre_up)
                mean_mean_post_up = np.mean(mean_post_up)
                mean_mean_pre_down = np.mean(mean_pre_down)
                mean_mean_post_down = np.mean(mean_post_down)
                
                sem_pre_up = np.std(mean_pre_up) / np.sqrt(len(mean_pre_up))
                sem_post_up = np.std(mean_post_up) / np.sqrt(len(mean_post_up))
                sem_pre_down = np.std(mean_pre_down) / np.sqrt(len(mean_pre_down))
                sem_post_down = np.std(mean_post_down) / np.sqrt(len(mean_post_down))
                
                t_pre, p_pre = ttest_ind(mean_pre_up, mean_pre_down)
                t_post, p_post = ttest_ind(mean_post_up, mean_post_down)
                significant_pre = p_pre < 0.05
                significant_post = p_post < 0.05
                significant = significant_pre or significant_post
                
                if significant:
                    if region == 'ACC':
                        acc_diff_neurons[roi] += 1
                    elif region == 'BLA':
                        bla_diff_neurons[roi] += 1
                
                ax = axes[i]
                bar_width = 0.35
                bars = ax.bar(['Pre Up', 'Pre Down', 'Post Up', 'Post Down'], 
                              [mean_mean_pre_up, mean_mean_pre_down, mean_mean_post_up, mean_mean_post_down],
                              yerr=[sem_pre_up, sem_pre_down, sem_post_up, sem_post_down], 
                              capsize=5, color=['blue', 'red', 'blue', 'red'])
                
                ax.set_title(f'ROI: {roi}')
                ax.set_xlabel('Condition')
                ax.set_ylabel('Mean Spike Count')
                if significant_pre:
                    ax.annotate('*', xy=(0.5, max(mean_mean_pre_up, mean_mean_pre_down) + max(sem_pre_up, sem_pre_down)), 
                                fontsize=20, ha='center', color='black')
                elif significant_post:
                    ax.annotate('*', xy=(2.5, max(mean_mean_post_up, mean_mean_post_down) + max(sem_post_up, sem_post_down)), 
                                fontsize=20, ha='center', color='black')
                ax.legend(bars, ['Pre Up', 'Pre Down', 'Post Up', 'Post Down'])
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(plot_dir, f'unit_{unit}_session_{session_name}_region_{region}_roi_response.png')
            plt.savefig(plot_path)
            plt.close(fig)
        except Exception as e:
            logging.error(f"Error processing unit {unit} in session {session_name}, region {region}: {e}")
            continue
    
    # Pie charts for ACC and BLA neurons
    for region, diff_neurons, total_neurons in [('ACC', acc_diff_neurons, acc_total_neurons), ('BLA', bla_diff_neurons, bla_total_neurons)]:
        try:
            fig, axes = plt.subplots(1, len(rois), figsize=(len(rois) * 5, 5))
            fig.suptitle(f'Differentiating Neurons by ROI ({region})')
            for i, roi in enumerate(rois):
                ax = axes[i]
                diff_count = diff_neurons[roi]
                ax.pie([diff_count, total_neurons - diff_count],
                       labels=['Differentiating', 'Non-Differentiating'],
                       autopct='%1.1f%%', startangle=140)
                ax.set_title(f'ROI: {roi}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pie_chart_path = os.path.join(plot_dir, f'{region.lower()}_roi_differentiating_neurons.png')
            plt.savefig(pie_chart_path)
            plt.close(fig)
        except Exception as e:
            logging.error(f"Error processing pie chart for region {region}: {e}")
            continue





