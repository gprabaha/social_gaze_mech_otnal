#!/bin/bash
#SBATCH --output=job_scripts/%x_%j.out
#SBATCH --error=job_scripts/%x_%j.err

module load miniconda
conda activate nn_gpu
sbatch job_scripts/dsq-joblist-2024-06-07.sh