#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:28:25 2024

Author: pg496
"""


from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
import multiprocessing
from multiprocessing import cpu_count, Pool
import numpy as np
import os
import scipy.signal as signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from tqdm import tqdm

import importlib
import logging
import pdb
import warnings
import gc
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the number of CPUs from the environment variable
slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
# If the environment variable is not set or empty, use cpu_count()
if slurm_cpus:
    num_cpus = int(slurm_cpus)
else:
    num_cpus = cpu_count()
print(f"Number of CPUs: {num_cpus}")


class ClusterFixationDetector:
    def __init__(self, samprate=1/1000, params=None):
        self.params = params
        self.samprate = samprate
        self.use_parallel = params['use_parallel']
        self.variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
        self.fltord = 60
        self.lowpasfrq = 30
        self.nyqfrq = 1000 / 2
        self.flt = signal.firwin2(self.fltord, [0, self.lowpasfrq / self.nyqfrq, self.lowpasfrq / self.nyqfrq, 1], [1, 1, 0, 0])
        self.buffer = int(100 / (self.samprate * 1000))
        self.fixationstats = []
        self.num_cpus = num_cpus


    def detect_fixations(self, eyedat):
        if not eyedat:
            raise ValueError("No data file found")
        self.fixationstats = self.process_eyedat(eyedat)
        return self.fixationstats


    def process_eyedat(self, data):
        if len(data[0]) > int(500 / (self.samprate * 1000)):
            x, y = self.preprocess_data(data)
            vel, accel, angle, dist, rot = self.extract_parameters(x, y)
            points = self.normalize_parameters(dist, vel, accel, rot)
            T, meanvalues, stdvalues = self.global_clustering(points)
            fixationcluster, fixationcluster2 = self.find_fixation_clusters(meanvalues, stdvalues)
            T = self.classify_fixations(T, fixationcluster, fixationcluster2)
            fixationindexes, fixationtimes = self.behavioral_index(T, 1)
            fixationtimes = self.apply_duration_threshold(fixationtimes, int(0.025 / self.samprate))
            notfixations = self.local_reclustering((fixationtimes, points))
            fixationindexes = self.remove_not_fixations(fixationindexes, notfixations)
            saccadeindexes, saccadetimes = self.classify_saccades(fixationindexes, points)
            fixationtimes, saccadetimes = self.round_times(fixationtimes, saccadetimes)
            pointfix, pointsac, recalc_meanvalues, recalc_stdvalues = self.calculate_cluster_values(fixationtimes, saccadetimes, data)

            return {
                'fixationtimes': fixationtimes * self.samprate,  # Convert indices to time points
                'fixations': self.extract_fixations(fixationtimes, data),
                'saccadetimes': saccadetimes * self.samprate,  # Convert indices to time points
                'FixationClusterValues': pointfix,
                'SaccadeClusterValues': pointsac,
                'MeanClusterValues': recalc_meanvalues,
                'STDClusterValues': recalc_stdvalues,
                'XY': np.array([data[0], data[1]]),
                'variables': self.variables
            }
        else:
            return {
                'fixationtimes': [],
                'fixations': [],
                'saccadetimes': [],
                'FixationClusterValues': [],
                'SaccadeClusterValues': [],
                'MeanClusterValues': [],
                'STDClusterValues': [],
                'XY': np.array([data[0], data[1]]),
                'variables': self.variables
            }


    def preprocess_data(self, eyedat):
        x = np.pad(eyedat[0], (self.buffer, self.buffer), 'reflect')
        y = np.pad(eyedat[1], (self.buffer, self.buffer), 'reflect')
        x = self.resample_data(x)
        y = self.resample_data(y)
        x = self.apply_filter(x)
        y = self.apply_filter(y)
        x = x[100:-100]
        y = y[100:-100]
        return x, y


    def resample_data(self, data):
        t_old = np.linspace(0, len(data) - 1, len(data))
        resample_factor = self.samprate * 1000
        if resample_factor > 1:
            print(f"Resample factor is too large: {resample_factor}")
            raise ValueError("Resample factor is too large, leading to excessive memory usage.")
        t_new = np.linspace(0, len(data) - 1, int(len(data) * resample_factor))
        f = interp1d(t_old, data, kind='linear')
        return f(t_new)


    def apply_filter(self, data):
        return signal.filtfilt(self.flt, 1, data)


    def extract_parameters(self, x, y):
        velx = np.diff(x)
        vely = np.diff(y)
        vel = np.sqrt(velx ** 2 + vely ** 2)
        accel = np.abs(np.diff(vel))
        angle = np.degrees(np.arctan2(vely, velx))
        vel = vel[:-1]
        rot = np.zeros(len(x) - 2)
        dist = np.zeros(len(x) - 2)
        for a in range(len(x) - 2):
            rot[a] = np.abs(angle[a] - angle[a + 1])
            dist[a] = np.sqrt((x[a] - x[a + 2]) ** 2 + (y[a] - y[a + 2]) ** 2)
        rot[rot > 180] -= 180
        rot = 360 - rot
        return vel, accel, angle, dist, rot


    def normalize_parameters(self, dist, vel, accel, rot):
        points = np.stack([dist, vel, accel, rot], axis=1)
        for ii in range(points.shape[1]):
            thresh = np.mean(points[:, ii]) + 3 * np.std(points[:, ii])
            points[points[:, ii] > thresh, ii] = thresh
            points[:, ii] = points[:, ii] - np.min(points[:, ii])
            points[:, ii] = points[:, ii] / np.max(points[:, ii])
        return points


    def global_clustering(self, points):
        logging.info("Starting global_clustering...")
        numclusts_range = list(range(2, 6))
        num_cpus = 4  # Replace this with the actual number of CPUs you have
        max_workers = min(len(numclusts_range), num_cpus)  # Limiting to available CPUs
        sil = np.zeros(5)
        results = []
        if self.use_parallel:
            logging.info("Using parallel processing for global clustering")
            results = self.parallel_global_clustering(points, numclusts_range, max_workers)
            gc.collect()
            time.sleep(1)  # Add a small delay to ensure cleanup
        else:
            logging.info("Using serial processing for global clustering")
            for numclusts in tqdm(numclusts_range, desc="Global Clustering Progress"):
                try:
                    result = self.cluster_and_silhouette(points, numclusts)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error processing numclusts {numclusts}: {e}")
        for numclusts, score in results:
            sil[numclusts - 2] = score
        numclusters = np.argmax(sil) + 2
        logging.info(f"Optimal number of clusters: {numclusters}")
        T = KMeans(n_clusters=numclusters, n_init=5).fit(points)
        labels = T.labels_
        meanvalues = np.array([np.mean(points[labels == i], axis=0) for i in range(numclusters)])
        stdvalues = np.array([np.std(points[labels == i], axis=0) for i in range(numclusters)])
        logging.info("Global clustering completed successfully")
        return labels, meanvalues, stdvalues


    def parallel_global_clustering(self, points, numclusts_range, max_workers):
        logger.info("Using parallel processing with multiprocessing.Pool")
        with Pool(processes=max_workers) as pool:
            args = [(points, numclusts) for numclusts in numclusts_range]
            results = []
            for result in tqdm(
                pool.imap(self.cluster_and_silhouette_wrapper, args),
                total=len(numclusts_range),
                desc="Global Clustering Progress"):
                try:
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
            pool.close()
            pool.join()
        return results


    def cluster_and_silhouette_wrapper(self, args):
        points, numclusts = args
        return self.cluster_and_silhouette(points, numclusts)


    def cluster_and_silhouette(self, points, numclusts):
        T = KMeans(n_clusters=numclusts, n_init=5).fit(points[::10, 1:4])
        silh = self.inter_vs_intra_dist(points[::10, 1:4], T.labels_)
        return numclusts, np.mean(silh)


    def find_fixation_clusters(self, meanvalues, stdvalues):
        fixationcluster = np.argmin(np.sum(meanvalues[:, 1:3], axis=1))
        fixationcluster2 = np.where(meanvalues[:, 1] < meanvalues[fixationcluster, 1] + 3 * stdvalues[fixationcluster, 1])[0]
        fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
        return fixationcluster, fixationcluster2


    def classify_fixations(self, T, fixationcluster, fixationcluster2):
        T[T == fixationcluster] = 100
        for cluster in fixationcluster2:
            T[T == cluster] = 100
        T[T != 100] = 2
        T[T == 100] = 1
        return T


    def behavioral_index(self, T, label):
        indexes = np.where(T == label)[0]
        return indexes, self.find_behavioral_times(indexes)


    def find_behavioral_times(self, indexes):
        dind = np.diff(indexes)
        gaps = np.where(dind > 1)[0]
        if gaps.size > 0:
            behaveind = np.split(indexes, gaps + 1)
        else:
            behaveind = [indexes]
        behaviortime = np.zeros((2, len(behaveind)), dtype=int)
        for i, ind in enumerate(behaveind):
            behaviortime[:, i] = [ind[0], ind[-1]]
        return behaviortime


    def apply_duration_threshold(self, times, threshold):
        return times[:, np.diff(times, axis=0)[0] >= threshold]


    def local_reclustering(self, data):
        logger.info("Starting local_reclustering...")
        fix_times, points = data
        notfixations = []
        if 0: # self.params['use_parallel']:
            importlib.reload(concurrent.futures)
            with concurrent.futures.ProcessPoolExecutor() as executor_2:
                logger.info("Parallelize local_reclustering over the detected fix time points")
                futures = {executor_2.submit(self.process_fixation_wrapper, fix, points): fix for fix in fix_times.T}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(fix_times.T), desc="Parallel Reclustering Progress"):
                    try:
                        notfixations.extend(future.result())
                    except Exception as e:
                        logger.exception("Exception occurred during parallel local reclustering")
            executor_2.shutdown(wait=True)
            gc.collect()  # Ensure garbage collection after executor shutdown
        else:
            logger.info("Serial local_reclustering over the detected fix time points")
                futures = {executor_2.submit(self.process_fixation_wrapper, fix, points): fix for fix in
            for fix in tqdm(fix_times.T, desc="Serial Reclustering Progress"):
                notfixations.extend(self.process_fixation_local_reclustering(fix, points))
        logger.info("Finished local_reclustering...")
        return np.array(notfixations)


    def process_fixation_wrapper(self, fix, points):
        return self.process_fixation_local_reclustering(fix, points)


    def process_fixation_local_reclustering(self, fix, points):
        try:
            logger.info("Starting process_fixation_local_reclustering...")
            logger.debug(f"Fixation indices: {fix}")
            altind = np.arange(fix[0] - 50, fix[1] + 50)
            altind = altind[(altind >= 0) & (altind < len(points))]
            POINTS = points[altind]
            logger.debug(f"Altind: {altind}")
            logger.debug(f"Number of points for reclustering: {len(POINTS)}")
            numclusts_range = range(1, 6)
            logger.debug("Parallelizing local reclustering over numclusts range")
            sil_results = [self.compute_sil((numclusts, POINTS)) for numclusts in numclusts_range]
            sil = np.zeros(5)
            for mean_sil, numclusts in sil_results:
                sil[numclusts - 1] = mean_sil
            logger.debug(f"Silhouette scores: {sil}")
            numclusters = np.argmax(sil) + 1
            logger.debug(f"Optimal number of clusters: {numclusters}")
            T = KMeans(n_clusters=numclusters, n_init=5).fit(POINTS)
            medianvalues = np.array([np.median(POINTS[T.labels_ == i], axis=0) for i in range(numclusters)])
            logger.debug(f"Median values: {medianvalues}")
            fixationcluster = np.argmin(np.sum(medianvalues[:, 1:3], axis=1))  # Velocity and acceleration
            logger.debug(f"Fixation cluster index: {fixationcluster}")
            fixation_indices = np.where(T.labels_ == fixationcluster)[0]
            # Check if the fixation cluster has no points
            if fixation_indices.size == 0:
                logger.warning("No points assigned to the fixationcluster")
                return np.array([])  # Return an empty array directly
            # Label the fixation points as 100 temporarily
            T.labels_[fixation_indices] = 100
            fixationcluster2 = np.where(
                (medianvalues[:, 1] < medianvalues[fixationcluster, 1] + 3 * np.std(POINTS[fixation_indices][:, 1])) &
                (medianvalues[:, 2] < medianvalues[fixationcluster, 2] + 3 * np.std(POINTS[fixation_indices][:, 2]))
            )[0]
            fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
            logger.debug(f"Additional fixation clusters: {fixationcluster2}")
            for cluster in fixationcluster2:
                cluster_indices = np.where(T.labels_ == cluster)[0]
                T.labels_[cluster_indices] = 100
            # Final relabeling
            T.labels_[T.labels_ != 100] = 2
            T.labels_[T.labels_ == 100] = 1
            logger.info("Reclustering completed successfully")
            return altind[T.labels_ == 2]
        except Exception as e:
            logger.exception("Exception occurred during fixation local reclustering")
            return np.array([])


    def compute_sil(self, data):
        try:
            numclusts, POINTS = data
            T = KMeans(n_clusters=numclusts, n_init=5).fit(POINTS[::5])
            silh = self.inter_vs_intra_dist(POINTS[::5], T.labels_)
            return np.nanmean(silh), numclusts
        except Exception as e:
            logger.exception("Exception occurred during silhouette computation")
            return 0.0, numclusts


    def remove_not_fixations(self, fixationindexes, notfixations):
        fixationindexes = np.setdiff1d(fixationindexes, notfixations)
        return fixationindexes


    def classify_saccades(self, fixationindexes, points):
        saccadeindexes = np.setdiff1d(np.arange(len(points)), fixationindexes)
        saccadetimes = self.find_behavioral_times(saccadeindexes)
        return saccadeindexes, saccadetimes


    def round_times(self, fixationtimes, saccadetimes):
        round5 = np.mod(fixationtimes, self.samprate * 1000)
        round5[0, round5[0] > 0] = self.samprate * 1000 - round5[0, round5[0] > 0]
        round5[1] = -round5[1]
        fixationtimes = np.round((fixationtimes + round5) / (self.samprate * 1000)).astype(int)
        fixationtimes[fixationtimes < 1] = 1
        round5 = np.mod(saccadetimes, self.samprate * 1000)
        round5[0] = -round5[0]
        round5[1, round5[1] > 0] = self.samprate * 1000 - round5[1, round5[1] > 0]
        saccadetimes = np.round((saccadetimes + round5) / (self.samprate * 1000)).astype(int)
        saccadetimes[saccadetimes < 1] = 1
        return fixationtimes, saccadetimes


    def calculate_cluster_values(self, fixationtimes, saccadetimes, eyedat):
        x = eyedat[0]
        y = eyedat[1]
        pointfix = [self.extract_variables(x[fix[0]:fix[1]], y[fix[0]:fix[1]]) for fix in fixationtimes.T]
        pointsac = [self.extract_variables(x[sac[0]:sac[1]], y[sac[0]:sac[1]]) for sac in saccadetimes.T]
        recalc_meanvalues = [np.nanmean(pointfix, axis=0), np.nanmean(pointsac, axis=0)]
        recalc_stdvalues = [np.nanstd(pointfix, axis=0), np.nanstd(pointsac, axis=0)]
        return pointfix, pointsac, recalc_meanvalues, recalc_stdvalues


    def extract_fixations(self, fixationtimes, eyedat):
        x = eyedat[0]
        y = eyedat[1]
        fixations = [np.mean([x[fix[0]:fix[1]], y[fix[0]:fix[1]]], axis=1) for fix in fixationtimes.T]
        return np.array(fixations)


    def extract_variables(self, xss, yss):
        # warnings.showwarning = self.custom_warning_handler
        if len(xss) < 3:
            return np.full(6, np.nan)
        vel = np.sqrt(np.diff(xss) ** 2 + np.diff(yss) ** 2) / self.samprate
        angle = np.degrees(np.arctan2(np.diff(yss), np.diff(xss)))
        accel = np.abs(np.diff(vel)) / self.samprate
        dist = [np.sqrt((xss[a] - xss[a + 2]) ** 2 + (yss[a] - yss[a + 2]) ** 2) for a in range(len(xss) - 2)]
        rot = [np.abs(angle[a] - angle[a + 1]) for a in range(len(xss) - 2)]
        rot = [r if r <= 180 else 360 - r for r in rot]
        # Reset warning handler to default after this function
        # warnings.showwarning = warnings._showwarnmsg_impl
        return [np.max(vel), np.max(accel), np.mean(dist), np.mean(vel), np.abs(np.mean(angle)), np.mean(rot)]


    def custom_warning_handler(self, message, category, filename, lineno, file=None, line=None):
        if category == RuntimeWarning and 'invalid value encountered in sqrt' in str(message):
            pdb.set_trace()  # This will trigger the debugger
        else:
            # Use the default warning handler for other warnings
            warnings.showwarning(message, category, filename, lineno, file, line)


    def inter_vs_intra_dist(self, X, labels):
        n = len(labels)
        k = len(np.unique(labels))
        count = np.bincount(labels)
        if k == 0 or n == 0:
            print("Error: Labels or data points are empty.")
            return np.full(n, 0.0)  # Return an array of zeros
        if np.any(count == 0):
            print("Error: One or more clusters have no members.")
            return np.full(n, 0.0)  # Return an array of zeros
        try:
            mbrs = (np.arange(k) == labels[:, None])
            avgDWithin = np.full(n, np.inf)
            avgDBetween = np.full((n, k), np.inf)
            for j in range(n):
                distj = np.sum((X - X[j]) ** 2, axis=1)
                for i in range(k):
                    if i == labels[j]:
                        if count[i] > 1:
                            avgDWithin[j] = np.sum(distj[mbrs[:, i]]) / (count[i] - 1)
                        else:
                            avgDWithin[j] = np.nan  # Avoid division by zero, set to NaN
                    else:
                        if count[i] > 0:
                            avgDBetween[j, i] = np.sum(distj[mbrs[:, i]]) / count[i]
                        else:
                            avgDBetween[j, i] = np.nan  # Avoid division by zero, set to NaN
            minavgDBetween = np.nanmin(avgDBetween, axis=1)  # Use nanmin to ignore NaNs
            # Ensure no division by zero
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for invalid operations
                silh = (minavgDBetween - avgDWithin) / np.maximum(avgDWithin, minavgDBetween)
                silh = np.nan_to_num(silh, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs and infs with 0
        except Exception as e:
            return np.full(n, 0.0)  # Return an array of zeros in case of an error
        return silh




# Example usage
if __name__ == "__main__":
    eyedat = [...]  # Your input data
    detector = ClusterFixationDetector()
    fixationstats = detector.detect_fixations(eyedat)
