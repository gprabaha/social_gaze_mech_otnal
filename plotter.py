#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:35:03 2024

@author: pg496
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from scipy.stats import ttest_ind
import seaborn as sns
from datetime import datetime
import logging
from matplotlib_venn import venn3
import random

from tqdm import tqdm

import util
import load_data

import pdb

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def plot_behavior_for_session(session, fixations_df, saccades_df, gaze_positions_list, plots_dir):
    session_fixations = fixations_df[fixations_df['session_name'] == session]
    session_saccades = saccades_df[saccades_df['session_name'] == session]
    gaze_data = [data for data in gaze_positions_list if data[1]['session_name'] == session][0]
    gaze_positions = gaze_data[0]
    info = gaze_data[1]
    mon_down_blocks = session_fixations[session_fixations['block'] == 'mon_down']['run'].unique()
    n_runs = len(mon_down_blocks)
    fig, axes = plt.subplots(n_runs + 1, 1, figsize=(20, 10 * (n_runs + 1)))
    # Plot for combined mon_up blocks
    mon_up_fixations = session_fixations[session_fixations['block'] == 'mon_up']
    mon_up_saccades = session_saccades[session_saccades['block'] == 'mon_up']
    plot_session_gaze_trajectory(axes[0], gaze_positions, mon_up_fixations, mon_up_saccades, info, 'mon_up blocks')
    # Plot for each mon_down run
    for i, run in enumerate(mon_down_blocks):
        run_fixations = session_fixations[(session_fixations['block'] == 'mon_down') & (session_fixations['run'] == run)]
        run_saccades = session_saccades[(session_saccades['block'] == 'mon_down') & (session_saccades['run'] == run)]
        plot_session_gaze_trajectory(axes[i + 1], gaze_positions, run_fixations, run_saccades, info, f'mon_down run {run}')
    fig.suptitle(f'Session: {session}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{session}_behavior.png'))
    plt.close(fig)


def plot_session_gaze_trajectory(ax, gaze_positions, fixations, saccades, info, title):
    ax.plot(gaze_positions[:, 0], gaze_positions[:, 1], color='gray', label='Gaze Trajectory')
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(fixations['start_index'].min(), fixations['end_index'].max())
    # Overlay fixations with color coding based on time
    for _, fixation in fixations.iterrows():
        fixation_positions = gaze_positions[fixation['start_index']: fixation['end_index']]
        color = cmap(norm(fixation['start_index']))
        ax.scatter(fixation_positions[:, 0], fixation_positions[:, 1], color=color, label='Fixations', s=10)
    # Overlay saccades with color coding based on time
    for _, saccade in saccades.iterrows():
        saccade_positions = gaze_positions[saccade['start_index']: saccade['end_index']]
        color = cmap(norm(saccade['start_index']))
        ax.plot(saccade_positions[:, 0], saccade_positions[:, 1], color=color, label='Saccades')
    # Overlay bounding boxes
    for key, bbox in info.items():
        if 'bbox' in key:
            bottom_left = bbox['bottomLeft']
            top_right = bbox['topRight']
            rect = plt.Rectangle(bottom_left, top_right[0] - bottom_left[0], top_right[1] - bottom_left[1], fill=False, edgecolor='yellow')
            ax.add_patch(rect)
    ax.set_title(title)
    ax.legend()
























def plot_fixation_proportions_for_diff_conditions(labelled_fixations, params):
    """
    Plots the proportion of fixations on different ROIs for different conditions.

    Parameters:
    - params (dict): Dictionary containing parameters.
    """
    root_data_dir = params['root_data_dir']
    if params.get('export_plots_to_local_folder', False):
        plots_dir = 'plots'
    else:
        plots_dir = os.path.join(root_data_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    remap_flag = util.get_filename_flag_info(params)
    fixation_labels_m1 = labelled_fixations
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
    plot_filename = f'fixation_proportions.png'
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
    if params.get('export_plots_to_local_folder', False):
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


def plot_fixation_heatmaps(labelled_fixations, params):
    root_data_dir = params['root_data_dir']
    if params.get('export_plots_to_local_folder', True):
        plots_dir = util.add_date_dir_to_path('plots/fix_heatmaps')
    else:
        plots_dir = util.add_date_dir_to_path(
            os.path.join(root_data_dir, 'plots', 'fix_heatmaps'))
    os.makedirs(plots_dir, exist_ok=True)
    labelled_gaze_positions_m1 = load_data.load_labelled_gaze_positions(
        params)
    all_fixation_labels = labelled_fixations
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
            pie_chart_path = os.path.join(
                plot_dir, f'{region.lower()}_roi_differentiating_neurons.png')
            plt.savefig(pie_chart_path)
            plt.close(fig)
        except Exception as e:
            logging.error(f"Error processing pie chart for region {region}: {e}")
            continue


def plot_unit_response_to_rois(unit, rois, pre_means, post_means, pre_errors, post_errors, significant_pre, significant_post, output_dir):
    try:
        def add_significance_lines(ax, x_positions, data_means, significant_matrix, y_offset=1.05):
            """
            Add lines and asterisks to denote significance between bars.
            """
            for i in range(len(x_positions)):
                for j in range(i + 1, len(x_positions)):
                    if significant_matrix[i, j]:
                        x1, x2 = x_positions[i], x_positions[j]
                        y = max(data_means[i], data_means[j]) * y_offset
                        ax.plot([x1, x1, x2, x2], [data_means[i], y, y, data_means[j]], color='black')
                        ax.text((x1 + x2) * 0.5, y, '*', ha='center')
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        x_pos = np.arange(len(rois))
        # Pre-fixation plot
        try:
            axes[0].bar(x_pos, pre_means, yerr=pre_errors, capsize=5, color='b', alpha=0.7)
            axes[0].set_title(f'Pre-Fixation Spike Count for Unit {unit}')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(rois)
            axes[0].set_ylabel('Mean Spike Count')
            add_significance_lines(axes[0], x_pos, pre_means, significant_pre)
        except Exception as e:
            logger.error(f"Error plotting pre-fixation data for unit {unit}: {e}")
        # Post-fixation plot
        try:
            axes[1].bar(x_pos, post_means, yerr=post_errors, capsize=5, color='r', alpha=0.7)
            axes[1].set_title(f'Post-Fixation Spike Count for Unit {unit}')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(rois)
            axes[1].set_ylabel('Mean Spike Count')
            add_significance_lines(axes[1], x_pos, post_means, significant_post)
        except Exception as e:
            logger.error(f"Error plotting post-fixation data for unit {unit}: {e}")
        fig.tight_layout()
        plt_path = os.path.join(output_dir, f'pre_and_post_fixation_response_to_roi_for_unit_{unit}.png')
        plt.savefig(plt_path)
        plt.close(fig) 
    except Exception as e:
        logger.error(f"Error in plot_unit_response_to_rois for unit {unit}: {e}")















def plot_roi_comparisons_for_unit(unit, region, pre_data, post_data, output_dir):
    plt.ioff()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Unit {unit} - Region {region}')
    comparisons = [
        ('eye_bbox', 'left_obj_bbox'),
        ('eye_bbox', 'right_obj_bbox'),
        ('eye_bbox', 'left_right_combined'),
        ('face_bbox', 'left_obj_bbox'),
        ('face_bbox', 'right_obj_bbox'),
        ('face_bbox', 'left_right_combined')
    ]
    colors = sns.color_palette("Set2", 2)
    for i, (roi1, roi2) in enumerate(comparisons):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        if roi2 == 'left_right_combined':
            pre_data_combined = np.concatenate((pre_data['left_obj_bbox'], pre_data['right_obj_bbox']), axis=0)
            post_data_combined = np.concatenate((post_data['left_obj_bbox'], post_data['right_obj_bbox']), axis=0)
        else:
            pre_data_combined = pre_data[roi2]
            post_data_combined = post_data[roi2]
        data = [
            pre_data[roi1].mean(axis=1).astype(float), pre_data_combined.mean(axis=1).astype(float),
            post_data[roi1].mean(axis=1).astype(float), post_data_combined.mean(axis=1).astype(float)
        ]
        labels = [
            'Pre ' + roi1, 'Pre ' + roi2, 'Post ' + roi1, 'Post ' + roi2
        ]
        sns.violinplot(data=data, ax=ax, palette=[colors[0], colors[0], colors[1], colors[1]])
        sns.stripplot(data=data, ax=ax, color='k', size=3, jitter=True)
        ax.set_title(f'{roi1} vs {roi2}')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean Spike Count")
    plt.savefig(os.path.join(output_dir, f'unit_{unit}_roi_comparison.png'))
    plt.close(fig)



def plot_pie_charts(region, results, output_base_dir):
    for comparison in results['both'].keys():
        total_units = len(set(results['pre'][comparison]) | set(results['post'][comparison]) | set(results['both'][comparison]) | set(results['neither'][comparison]))
        significant_pre = len(set(results['pre'][comparison]))
        significant_post = len(set(results['post'][comparison]))
        significant_both = len(set(results['both'][comparison]))
        significant_neither = len(set(results['neither'][comparison]))

        sizes = [
            significant_neither,
            significant_pre,
            significant_post,
            significant_both
        ]
        labels = ['Neither', 'Just Pre', 'Just Post', 'Both Pre and Post']
        colors = sns.color_palette("Set2", 4)

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Significant Differences in {region} - {comparison}')
        plt.savefig(os.path.join(output_base_dir, f'{region}_{comparison}_significant_pie_chart.png'))
        plt.close(fig)

def plot_venn_diagrams(region, results, output_base_dir):
    comparisons = [
        ('eye_bbox', 'left_obj_bbox'),
        ('eye_bbox', 'right_obj_bbox'),
        ('eye_bbox', 'left_right_combined'),
        ('face_bbox', 'left_obj_bbox'),
        ('face_bbox', 'right_obj_bbox'),
        ('face_bbox', 'left_right_combined')
    ]

    for roi1, roi2 in comparisons:
        comparison = roi1 + " vs " + roi2
        pre_only = set(results['pre'][comparison])
        post_only = set(results['post'][comparison])
        both = set(results['both'][comparison])
        either = set(results['either'][comparison])
        neither = set(results['neither'][comparison])
        total_units = len(pre_only | post_only | both | neither)

        fig, ax = plt.subplots()
        venn3(subsets=(
                len(neither),  # A: Neither
                len(pre_only) - len(pre_only & post_only),  # B: Just Pre
                len(pre_only & post_only),  # AB: Both
                len(post_only) - len(pre_only & post_only),  # C: Just Post
                len(pre_only & neither),  # AC: Pre & Neither
                len(post_only & neither),  # BC: Post & Neither
                len(pre_only & post_only & neither)  # ABC: Pre, Post & Neither
            ), set_labels=('Neither', 'Just Pre', 'Just Post'))

        plt.title(f'Significant Differences in {region} - {comparison}')
        plt.annotate(f'Total units considered: {total_units}', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=12)
        plt.savefig(os.path.join(output_base_dir, f'{region}_{comparison}_significant_venn_diagram.png'))
        plt.close(fig)


















