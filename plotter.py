#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:35:03 2024

@author: pg496
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
import os

import util
import load_data

import pdb


def plot_fixation_proportions_for_diff_conditions(fixation_labels_m1, params):
    # Extract the root_data_dir from params
    root_data_dir = params['root_data_dir']
    # Get the filename flag info using the util function
    remap_flag = util.get_filename_flag_info(params)
    # Ensure the plots directory exists
    plots_dir = os.path.join(root_data_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    # Filter out rows where 'run' is NaN
    valid_runs = fixation_labels_m1[~fixation_labels_m1['run'].isna()]
    # Define the conditions
    conditions = {
        'mon_up': valid_runs['block'] == 'mon_up',
        'mon_down': valid_runs['block'] == 'mon_down'
    }
    agents = {
        'Lynch': valid_runs['agent'] == 'Lynch',
        'Tarantino': valid_runs['agent'] == 'Tarantino'
    }
    rois = ['face_bbox', 'eye_bbox', 'left_obj_bbox', 'right_obj_bbox']
    # Initialize the plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(f'Proportion of Fixations on Different ROIs{remap_flag}', fontsize=16)
    # Iterate over agents and conditions to create subplots
    for i, (agent_name, agent_cond) in enumerate(agents.items()):
        for j, (block_name, block_cond) in enumerate(conditions.items()):
            ax = axes[i, j]
            # Filter data for current condition and agent
            data = valid_runs[agent_cond & block_cond]
            # Calculate proportions
            proportions = [np.mean(data['fix_roi'] == roi) for roi in rois]
            # Bar plot
            ax.bar(rois, proportions, color=['blue', 'orange', 'green', 'red'])
            ax.set_title(f'{agent_name} - {block_name}')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Proportion of Fixations')
            ax.set_xlabel('ROI')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save the plot
    plot_filename = f'fixation_proportions{remap_flag}.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()


def plot_gaze_heatmaps_for_conditions(params):
    root_data_dir = params['root_data_dir']
    heatmap_base_dir = os.path.join(root_data_dir, 'plots', 'heatmap')
    os.makedirs(heatmap_base_dir, exist_ok=True)
    conditions = [(roi, gaze) for roi in [True, False] for gaze in [True, False]]
    for roi_condition, gaze_condition in conditions:
        # Update params for current condition
        params['map_roi_coord_to_eyelink_space'] = roi_condition
        params['map_gaze_pos_coord_to_eyelink_space'] = gaze_condition
        # Load the corresponding labelled gaze positions
        labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(params)
        # Create a directory for this condition
        condition_dir = f"roi_{roi_condition}_gaze_{gaze_condition}"
        plots_dir = os.path.join(heatmap_base_dir, condition_dir)
        os.makedirs(plots_dir, exist_ok=True)
        # Generate the plots
        plot_gaze_heatmaps_for_all_sessions(labelled_gaze_positions_m1, params, plots_dir)


def plot_gaze_heatmaps_for_all_sessions(labelled_gaze_positions_m1, params, plots_dir):
    print(f"\nCondition: Gaze remap -- {params['map_gaze_pos_coord_to_eyelink_space']}| ROI remap -- {params['map_roi_coord_to_eyelink_space']}\n")
    for session_idx, (gaze_positions, session_info) in enumerate(tqdm(labelled_gaze_positions_m1, desc="Processing Sessions")):
        plot_gaze_heatmap_for_one_session(gaze_positions, session_info, session_idx, plots_dir)


def plot_gaze_heatmap_for_one_session(gaze_positions, session_info, session_idx, plots_dir):
    sampling_rate = session_info['sampling_rate']
    start_times = session_info['startS']
    stop_times = session_info['stopS']
    roi_bb_corners = session_info['roi_bb_corners']
    for run_idx, (start, stop) in enumerate(zip(start_times, stop_times)):
        start_idx = round(start / sampling_rate)
        stop_idx = round(stop / sampling_rate)
        # Isolate the gaze positions within the start and stop indices
        gaze_run = gaze_positions[start_idx:stop_idx]
        # Generate the 2D histogram
        heatmap, xedges, yedges = np.histogram2d(gaze_run[:, 0], gaze_run[:, 1], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
        # Plot ROI quadrilaterals
        for roi_name, corners in roi_bb_corners.items():
            points = [corners['topRight'], corners['topLeft'], corners['bottomLeft'], corners['bottomRight']]
            polygon = Polygon(points, closed=True, fill=False, edgecolor='blue', linewidth=2, label=roi_name)
            plt.gca().add_patch(polygon)
        plt.colorbar(label='Frequency')
        plt.title(f'Session {session_idx + 1}, Run {run_idx + 1}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend(loc='upper right')
        # Save the plot
        plot_filename = f'session_{session_idx + 1}_run_{run_idx + 1}.png'
        plt.savefig(os.path.join(plots_dir, plot_filename))
        plt.close()






