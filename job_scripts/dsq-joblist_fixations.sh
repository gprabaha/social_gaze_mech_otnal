#!/bin/bash
#SBATCH --output ./job_scripts/
#SBATCH --array 0-30
#SBATCH --job-name dsq-fixation_joblist
#SBATCH --partition psych_day --cpus-per-task 8 --mem-per-cpu 4096 -t 2:00:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/fixation_joblist.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts

