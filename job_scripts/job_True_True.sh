#!/bin/bash
#SBATCH --job-name=test_comb_True_True
#SBATCH --output=job_scripts/output_True_True.txt
#SBATCH --error=job_scripts/error_True_True.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --partition=psych_day
#SBATCH --mail-type=FAIL

module load miniconda
conda activate nn_gpu
export map_roi_coord_to_eyelink_space=True
export map_gaze_pos_coord_to_eyelink_space=True

python test_coord_mapping_to_eyelink_space.py
