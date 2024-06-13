#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:22:45 2024

@author: pg496
"""

# process_session.py
import os
import argparse
import pandas as pd
import logging

import util
import load_data
import curate_data

def main():
    parser = argparse.ArgumentParser(description='Process a single session to generate raster data.')
    parser.add_argument('--session', required=True, help='Path to the session to process')
    args = parser.parse_args()
    session_name = os.path.basename(os.path.normpath(args.session))
    
    params = util.get_params()
    root_data_dir, params = util.fetch_root_data_dir(params)
    data_source_dir, params = util.fetch_data_source_dir(params)
    session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
    processed_data_dir, params = util.fetch_processed_data_dir(params)

    # Load fixation and spiketimes data (assuming these functions are available and correct)
    labelled_fixations = load_data.load_m1_fixation_labels(params)
    labelled_spiketimes = load_data.load_processed_spiketimes(params)
    
    # Filter data for the specific session
    labelled_fixations = labelled_fixations[labelled_fixations['session_name'] == session_name]
    labelled_spiketimes = labelled_spiketimes[labelled_spiketimes['session_name'] == session_name]

    # Load parameters (assuming this function or equivalent is available)

    # Generate raster data for the session
    raster_data = curate_data.generate_session_raster(session_name, labelled_fixations, labelled_spiketimes, params)
    
    # Save the generated raster data
    curate_data.save_labelled_fixation_rasters(raster_data, params)

if __name__ == '__main__':
    main()
