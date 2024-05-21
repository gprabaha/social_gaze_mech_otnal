#!/bin/bash

# Set common parameters
CPUS_PER_TASK=12
MEMORY_PER_CPU=16G
PARTITION=psych_day
TIME_LIMIT=4:00:00
SCRIPT=analyze_gaze_signals_cluster.py

# Create job_scripts directory if it doesn't exist
JOB_SCRIPTS_DIR="job_scripts"
mkdir -p $JOB_SCRIPTS_DIR

# Define combinations of flags
combinations=(
    "False False"
    "False True"
    "True False"
    "True True"
)

# Submit a job for each combination
for combination in "${combinations[@]}"; do
    IFS=' ' read -r map_roi map_gaze <<< "$combination"
    
    # Create a job script
    JOB_SCRIPT="${JOB_SCRIPTS_DIR}/job_${map_roi}_${map_gaze}.sh"
    cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=analyze_gaze_${map_roi}_${map_gaze}
#SBATCH --output=${JOB_SCRIPTS_DIR}/output_${map_roi}_${map_gaze}.txt
#SBATCH --error=${JOB_SCRIPTS_DIR}/error_${map_roi}_${map_gaze}.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem-per-cpu=${MEMORY_PER_CPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --mail-type=FAIL

module load miniconda
conda activate nn_gpu
export map_roi_coord_to_eyelink_space=${map_roi}
export map_gaze_pos_coord_to_eyelink_space=${map_gaze}

python ${SCRIPT}
EOT
    
    # Submit the job
    sbatch $JOB_SCRIPT
done
