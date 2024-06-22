#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:22:45 2024

@author: pg496
"""

import os
import argparse
import util
import load_data
from raster import RasterManager

def main():
    parser = argparse.ArgumentParser(description='Process a single session to generate raster data.')
    parser.add_argument('--session', required=True, help='Path to the session to process')
    args = parser.parse_args()
    session_path = args.session
    
    params = util.get_params()
    root_data_dir, params = util.fetch_root_data_dir(params)
    data_source_dir, params = util.fetch_data_source_dir(params)
    session_paths, params = util.fetch_session_subfolder_paths_from_source(params)
    processed_data_dir, params = util.fetch_processed_data_dir(params)

    # Load fixation and spiketimes data
    labelled_fixations = load_data.load_m1_fixation_labels(params)
    labelled_spiketimes = load_data.load_processed_spiketimes(params)

    # Instantiate RasterManager and generate the session raster
    raster_manager = RasterManager(params)
    raster_manager.generate_session_raster(session_path, labelled_fixations, labelled_spiketimes)

if __name__ == '__main__':
    main()
