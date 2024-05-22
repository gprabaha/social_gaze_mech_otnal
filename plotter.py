#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:35:03 2024

@author: pg496
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import util

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

# Example usage:
# plot_fixation_proportions(fixation_labels_m1)
