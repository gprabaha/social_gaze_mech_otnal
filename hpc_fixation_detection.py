#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:06:29 2024

Author: pg496
"""

import os
import subprocess
import logging
import time
import pickle
import json


class HPCFixationDetection:
    def __init__(self, params):
        self.params = params
        self.job_script_out_dir = './job_scripts/'

    def serialize_params(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.params, f)

    def generate_fixation_job_file(self, labelled_gaze_positions):
        job_file_path = os.path.join(self.job_script_out_dir, 'fixation_joblist.txt')
        os.makedirs(self.job_script_out_dir, exist_ok=True)

        # Use params['processed_data_dir'] for saving the parameters
        params_file_path = os.path.join(self.params['processed_data_dir'], 'params.json')
        self.serialize_params(params_file_path)
        
        with open(job_file_path, 'w') as file:
            for idx in range(len(labelled_gaze_positions)):
                command = (
                    "module load miniconda; "
                    "conda init bash; "
                    "conda activate gaze_processing; "
                    f"python process_session_fixations.py --session_index {idx} --params_file {params_file_path}"
                )
                file.write(command + "\n")
                
        # Check the contents of the job file for debugging
        with open(job_file_path, 'r') as file:
            job_commands = file.readlines()
            logging.info(f"Job commands in the job list: {len(job_commands)}")
            for command in job_commands:
                logging.debug(command.strip())
                
        return job_file_path

    def submit_job_array(self, job_file_path):
        try:
            job_script_path = os.path.join(self.job_script_out_dir, 'dsq-joblist_fixations.sh')
            subprocess.run(
                f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} -o {self.job_script_out_dir} --status-dir {self.job_script_out_dir} --partition psych_day --cpus-per-task 8 --mem-per-cpu 4096 -t 8:00:00 --mail-type FAIL',
                shell=True, check=True, executable='/bin/bash'
            )
            logging.info("Successfully generated the dSQ job script")
            if not os.path.isfile(job_script_path):
                logging.error(f"No job script found at {job_script_path}.")
                return
            logging.info(f"Using dSQ job script: {job_script_path}")
            
            result = subprocess.run(
                f'sbatch --job-name=fixation_jobs_dsq --output={self.job_script_out_dir}/fixation_session_%a.out --error={self.job_script_out_dir}/fixation_session_%a.err {job_script_path}',
                shell=True, check=True, capture_output=True, text=True, executable='/bin/bash'
            )
            logging.info(f"Successfully submitted jobs using sbatch for script {job_script_path}")
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"Submitted job array with ID: {job_id}")
            self.track_job_progress(job_id)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error during job submission process: {e}")
            raise

    def track_job_progress(self, job_id):
        logging.info(f"Tracking progress of job array with ID: {job_id}")
        while True:
            result = subprocess.run(
                f'squeue --job {job_id} -h -o %T',
                shell=True, capture_output=True, text=True, executable='/bin/bash'
            )
            if result.returncode != 0:
                logging.error(f"Error checking job status for job ID {job_id}: {result.stderr.strip()}")
                break
            job_statuses = result.stdout.strip().split()
            if not job_statuses:
                logging.info(f"Job array {job_id} has completed.")
                break
            running_jobs = [status for status in ('PENDING', 'RUNNING', 'CONFIGURING') if status in job_statuses]
            if not running_jobs:
                logging.info(f"Job array {job_id} has completed.")
                break
            else:
                logging.info(f"Job array {job_id} is still running. Checking again in 30 seconds...")
                time.sleep(30)

