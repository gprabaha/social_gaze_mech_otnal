#!/bin/bash
#SBATCH --output /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/
#SBATCH --array 0-31
#SBATCH --job-name dsq-raster_joblist
#SBATCH --mem-per-cpu 6g -t 02:00:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/raster_joblist.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts

