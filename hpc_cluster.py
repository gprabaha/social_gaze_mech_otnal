#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:57:11 2024

@author: pg496
"""

import os
import subprocess
import logging
import time

import pdb


def generate_job_file(session_paths):
    job_file_path = '/gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/raster_joblist.txt'
    os.makedirs('job_scripts', exist_ok=True)
    with open(job_file_path, 'w') as file:
        for session_path in session_paths:
            command = f"module load miniconda; conda init bash; conda activate nn_gpu; python process_session_raster.py --session {session_path}"
            file.write(command + "\n")
    return job_file_path


def submit_job_array(job_file_path):
    try:
        output_dir = '/gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/'
        job_script_path = os.path.join(output_dir, 'dsq-joblist_raster.sh')
        # Run the command to generate the job script
        subprocess.run(
            f'module load dSQ; dsq --job-file {job_file_path} --batch-file {job_script_path} -o {output_dir} --status-dir {output_dir} --cpus-per-task 4 --mem-per-cpu 16g -t 02:00:00 --mail-type FAIL',
            shell=True, check=True, executable='/bin/bash'
        )
        logging.info("Successfully generated the dSQ job script")
        if not os.path.isfile(job_script_path):
            logging.error(f"No job script found at {job_script_path}.")
            return
        logging.info(f"Using dSQ job script: {job_script_path}")
        pdb.set_trace()
        # Submit the job script with sbatch and ensure output is directed to the job_scripts directory
        result = subprocess.run(
            f'sbatch --output={output_dir}/%x_%A_%a.out --error={output_dir}/%x_%A_%a.err {job_script_path}',
            shell=True, check=True, capture_output=True, text=True, executable='/bin/bash'
        )
        logging.info(f"Successfully submitted jobs using sbatch for script {job_script_path}")
        # Extract the job ID from the sbatch output
        job_id = result.stdout.strip().split()[-1]
        logging.info(f"Submitted job array with ID: {job_id}")
        # Track the job progress
        track_job_progress(job_id)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during job submission process: {e}")
        raise


def track_job_progress(job_id):
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
        running_jobs = [status for status in job_statuses if status in ('PENDING', 'RUNNING', 'CONFIGURING')]
        if not running_jobs:
            logging.info(f"Job array {job_id} has completed.")
            break
        else:
            logging.info(f"Job array {job_id} is still running. Checking again in 30 seconds...")
            time.sleep(30)

