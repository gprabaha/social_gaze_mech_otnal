#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:25:48 2024

@author: pg496
"""

import os
import random
import glob
from scipy.io import loadmat

def get_subfolders(root_dir):
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    return subfolders

def extract_meta_info(session_paths):
    meta_info_list = []
    for session_path in session_paths:
        meta_info = {}
        file_list_info = glob.glob(f"{session_path}/*metaInfo.mat")
        file_list_runs = glob.glob(f"{session_path}/*runs.mat")

        if len(file_list_info) == 1:
            file_path_info = file_list_info[0]
            try:
                data_info = loadmat(file_path_info)
                info = data_info.get('info', None)
                # Selecting just the first run
                info = info[0][0]
                if info is not None:
                    monkey_1 = info['monkey_1'][0]
                    monkey_2 = info['monkey_2'][0]
                    OT_dose = float(info['OT_dose'][0])
                    NAL_dose = float(info['NAL_dose'][0])
                    meta_info.update({'monkey_1': monkey_1, 'monkey_2': monkey_2, 'OT_dose': OT_dose, 'NAL_dose': NAL_dose})
                else:
                    meta_info.update({'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None})
            except Exception as e:
                print(f"Error loading meta_info for session {session_path}: {e}")
                meta_info.update({'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None})
        else:
            print(f"Error: More than one metaInfo file found in session {session_path}, or no file found.")
            meta_info.update({'monkey_1': None, 'monkey_2': None, 'OT_dose': None, 'NAL_dose': None})

        if len(file_list_runs) == 1:
            file_path_runs = file_list_runs[0]
            try:
                data_runs = loadmat(file_path_runs)
                runs = data_runs.get('runs', None)
                if runs is not None:
                    startS = [run['startS'][0][0] for run in runs[0]]
                    stopS = [run['stopS'][0][0] for run in runs[0]]
                    num_runs = len(startS)
                    meta_info.update({'startS': startS, 'stopS': stopS, 'num_runs': num_runs})
                else:
                    meta_info.update({'startS': None, 'stopS': None, 'num_runs': 0})
            except Exception as e:
                print(f"Error loading runs for session {session_path}: {e}")
                meta_info.update({'startS': None, 'stopS': None, 'num_runs': 0})
        else:
            print(f"Error: More than one runs file found in session {session_path}, or no file found.")
            meta_info.update({'startS': None, 'stopS': None, 'num_runs': 0})

        meta_info_list.append(meta_info)

    return meta_info_list


############

root_data_dir = "/gpfs/milgram/project/chang/pg496/data_dir/otnal/"

session_paths = get_subfolders(root_data_dir)

meta_info_list = extract_meta_info(session_paths)

otnal_doses = [[meta_info['OT_dose'], meta_info['NAL_dose']] for meta_info in meta_info_list]

