#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:36:36 2024

@author: prabaha
"""

import numpy as np
from util import *

##################################################################
def find_saccades(x, y, sr, vel_thresh, min_samples, smooth_func):
    """
    Find start and stop indices of saccades.

    Parameters:
    - x: array-like, x-coordinates of eye movements
    - y: array-like, y-coordinates of eye movements
    - sr: float, sampling rate
    - vel_thresh: float, minimum velocity threshold for saccade onset
    - min_samples: int, minimum duration of a saccade in samples
    - smooth_func: function, function for smoothing input data

    Returns:
    - start_stops: list of arrays, start and stop indices of saccades for each trial
    """
    assert x.shape == y.shape
    num_trials = x.shape[0]
    
    start_stops = []

    for i in range(num_trials):
        x0 = smooth_func(x[i, :])
        y0 = smooth_func(y[i, :])
        
        vx = np.gradient(x0) * sr
        vy = np.gradient(y0) * sr
        
        vel_norm = np.sqrt(vx**2 + vy**2)  # Norm of velocity vector
        
        above_thresh = vel_norm >= vel_thresh
        
        starts, stops = find_vel_stops(vel_norm, above_thresh)
        
        durs = stops - starts
        within_thresh = durs >= min_samples
        starts = starts[within_thresh]
        stops = stops[within_thresh]
        
        starts, stops = merge_intervals(starts, stops)
        peak_velocities = peak_velocity(vel_norm, starts, stops)
        
        start_stops.append(np.column_stack((starts, stops, peak_velocities)))

    return start_stops