#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:37:42 2024

Author: pg496
"""

import os
import argparse
import logging
import json
import pickle
import numexpr as ne
import multiprocessing

import fix_and_saccades
import load_data


def main(session_index, params_file, num_cpus):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    # Set the maximum number of threads for NumExpr
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus)
    ne.set_num_threads(num_cpus)
    logger.info(f"NumExpr set to use {num_cpus} threads")
    # Load parameters from the JSON file
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
    logger.info(f"Starting fixation detection for session index: {session_index}")
    # Load labelled gaze positions
    labelled_gaze_positions, _ = load_data.load_labelled_gaze_positions(params)
    # Prepare session data for the specific index
    session_data = labelled_gaze_positions[session_index]
    session_data = (session_data[0], session_data[1], params)  # Prepare session data as needed by the function
    # Extract fixations and saccades using get_session_fixations_and_saccades
    fix_timepos_df, info, saccades_df = fix_and_saccades.get_session_fixations_and_saccades(session_data, num_cpus)
    # Save results
    output_dir = params['processed_data_dir']
    fixations_file = os.path.join(output_dir, f"{session_index}_fixations.pkl")
    with open(fixations_file, 'wb') as f:
        pickle.dump((fix_timepos_df, info, saccades_df), f)
    logger.info(f"Fixation detection completed for session index: {session_index}")
    logger.info(f"Results saved to: {fixations_file}")


if __name__ == "__main__":
    try:
        slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
        num_cpus = int(slurm_cpus)
        print(f"SLURM detected {num_cpus} CPUs")
    except Exception as e:
        print(f"Failed to detect cores with SLURM_CPUS_ON_NODE: {e}")
        num_cpus = None
    # If SLURM detection fails, fallback to multiprocessing.cpu_count()
    if num_cpus is None or num_cpus <= 1:
        num_cpus = multiprocessing.cpu_count()
        print(f"Multiprocessing detected {num_cpus} CPUs")
    parser = argparse.ArgumentParser(description="Process session fixation detection")
    parser.add_argument('--session_index', type=int, required=True, help='Index of the session in labelled gaze positions list')
    parser.add_argument('--params_file', type=str, required=True, help='Path to the JSON file with parameters')
    args = parser.parse_args()
    main(args.session_index, args.params_file, num_cpus)

