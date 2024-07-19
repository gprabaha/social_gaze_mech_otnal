#!/bin/bash
module load miniconda
conda deactivate
conda activate gaze_processing
python analyze_gaze_signals.py
