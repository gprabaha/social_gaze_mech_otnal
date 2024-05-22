#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:35:03 2024

@author: pg496
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pdb


def plot_fixation_proportions_for_diff_conditions(fixation_labels):
    # Filter out rows where 'run' is NaN
    valid_runs = fixation_labels[~fixation_labels['run'].isna()]
    valid_runs = fixation_labels
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
    fig.suptitle('Proportion of Fixations on Different ROIs', fontsize=16)
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
    plt.show()

# Example usage:
# plot_fixation_proportions(fixation_labels_m1)
