#!/bin/bash
#SBATCH --output dsq-joblist-%A_%2a-%N.out
#SBATCH --array 0-31
#SBATCH --job-name /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts//dsq-joblist
#SBATCH -O /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/ --mem-per-cpu 6g -t 02:00:00 --mail-type FAIL

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts/joblist.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/social_gaze_mech_otnal/job_scripts

