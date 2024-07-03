#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:28:25 2024

Author: pg496
"""


from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from tqdm import tqdm

import pdb

# Check number of available CPUs
num_cpus = cpu_count()
print(f"Number of available CPUs: {num_cpus}")


class ClusterFixationDetector:
    def __init__(self, samprate=1/1000, use_parallel=True):
        self.samprate = samprate
        self.use_parallel = use_parallel
        self.variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
        self.fltord = 60
        self.lowpasfrq = 30
        self.nyqfrq = 1000 / 2
        self.flt = signal.firwin2(self.fltord, [0, self.lowpasfrq / self.nyqfrq, self.lowpasfrq / self.nyqfrq, 1], [1, 1, 0, 0])
        self.buffer = int(100 / (self.samprate * 1000))
        self.fixationstats = []
        self.num_cpus = cpu_count()


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
            fixationtimes = self.apply_duration_threshold(fixationtimes, 25)

            notfixations = self.local_reclustering((fixationtimes, points))
            fixationindexes = self.remove_not_fixations(fixationindexes, notfixations)
            saccadeindexes, saccadetimes = self.classify_saccades(fixationindexes, points)

            fixationtimes, saccadetimes = self.round_times(fixationtimes, saccadetimes)

            pointfix, pointsac, recalc_meanvalues, recalc_stdvalues = self.calculate_cluster_values(fixationtimes, saccadetimes, data)

            return {
                'fixationtimes': fixationtimes,
                'fixations': self.extract_fixations(fixationtimes, data),
                'saccadetimes': saccadetimes,
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
        print("Starting global_clustering...")
        numclusts_range = list(range(2, 6))
        max_workers = min(len(numclusts_range), self.num_cpus)  # Limiting to 4 parallel jobs
        sil = np.zeros(5)
        if self.use_parallel:
            print("Using parallel processing with ProcessPoolExecutor")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(self.cluster_and_silhouette,
                                                 [(points, numclusts) for numclusts in numclusts_range]),
                                                 total=len(numclusts_range),
                                                 desc="Global Clustering Progress"))
            for numclusts, score in results:
                sil[numclusts - 2] = score
                print(f"Processed numclusts {numclusts}: Silhouette score = {score}")
        else:
            print("Using serial processing")
            for numclusts in tqdm(range(2, 6), desc="Global Clustering Progress"):
                try:
                    numclusts, score = self.cluster_and_silhouette((points, numclusts))
                    sil[numclusts - 2] = score
                    print(f"Processed numclusts {numclusts}: Silhouette score = {score}")
                except Exception as e:
                    print(f"Error processing numclusts {numclusts}: {e}")
        numclusters = np.argmax(sil) + 2
        print(f"Optimal number of clusters: {numclusters}")
        T = KMeans(n_clusters=numclusters, n_init=5).fit(points)
        labels = T.labels_
        meanvalues = np.array([np.mean(points[labels == i], axis=0) for i in range(numclusters)])
        stdvalues = np.array([np.std(points[labels == i], axis=0) for i in range(numclusters)])
        print("Global clustering completed successfully")
        return labels, meanvalues, stdvalues


    def cluster_and_silhouette(self, data):
        points, numclusts = data
        print(f'Doing kMeans for {numclusts} clusters now')
        T = KMeans(n_clusters=numclusts, n_init=5).fit(points[::10, 1:4])
        silh = self.inter_vs_intra_dist(points[::10, 1:4], T.labels_)
        print(f'kMeans for {numclusts} clusters done')
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
        print("Starting local_reclustering...")
        fix_times, points = data
        notfixations = []
        for fix in fix_times.T:
            altind = np.arange(fix[0] - 50, fix[1] + 50)
            altind = altind[(altind >= 0) & (altind < len(points))]
            POINTS = points[altind]
            numclusts_range = range(1, 6)
            max_workers = min(len(numclusts_range), self.num_cpus)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                sil_results = list(tqdm(executor.map(self.compute_sil,
                                                    [(numclusts, POINTS) for numclusts in numclusts_range]),
                                                    total=5,
                                                    desc="Local Reclustering Progress"))
            sil = np.zeros(5)
            for mean_sil, numclusts in sil_results:
                sil[numclusts - 1] = mean_sil
            numclusters = np.argmax(sil) + 1
            T = KMeans(n_clusters=numclusters, n_init=5).fit(POINTS)
            medianvalues = np.array([np.median(POINTS[T.labels_ == i], axis=0) for i in range(numclusters)])
            fixationcluster = np.argmin(np.sum(medianvalues[:, 1:3], axis=1))
            T.labels_[T.labels_ == fixationcluster] = 100
            fixationcluster2 = np.where((medianvalues[:, 1] < medianvalues[fixationcluster, 1] +
                                        3 * np.std(POINTS[T.labels_ == fixationcluster][:, 1])) &
                                        (medianvalues[:, 2] < medianvalues[fixationcluster, 2] +
                                        3 * np.std(POINTS[T.labels_ == fixationcluster][:, 2])))[0]
            fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
            for cluster in fixationcluster2:
                T.labels_[T.labels_ == cluster] = 100 
            T.labels_[T.labels_ != 100] = 2
            T.labels_[T.labels_ == 100] = 1
            notfixations.extend(altind[T.labels_ == 2])
        print("Finished local_reclustering...")
        return np.array(notfixations)


    def compute_sil(self, data):
        numclusts, POINTS = data
        T = KMeans(n_clusters=numclusts, n_init=5).fit(POINTS[::5])
        silh = self.inter_vs_intra_dist(POINTS[::5], T.labels_)
        return np.mean(silh), numclusts

    
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
        if len(xss) < 3:
            return np.full(6, np.nan)
        vel = np.sqrt(np.diff(xss) ** 2 + np.diff(yss) ** 2) / self.samprate
        angle = np.degrees(np.arctan2(np.diff(yss), np.diff(xss)))
        accel = np.abs(np.diff(vel)) / self.samprate
        dist = [np.sqrt((xss[a] - xss[a + 2]) ** 2 + (yss[a] - yss[a + 2]) ** 2) for a in range(len(xss) - 2)]
        rot = [np.abs(angle[a] - angle[a + 1]) for a in range(len(xss) - 2)]
        rot = [r if r <= 180 else 360 - r for r in rot]
        return [np.max(vel), np.max(accel), np.mean(dist), np.mean(vel), np.abs(np.mean(angle)), np.mean(rot)]


    def inter_vs_intra_dist(self, X, labels):
        n = len(labels)
        k = len(np.unique(labels))
        count = np.bincount(labels)
        mbrs = (np.arange(k) == labels[:, None])
        avgDWithin = np.full(n, np.inf)
        avgDBetween = np.full((n, k), np.inf)
        for j in range(n):
            distj = np.sum((X - X[j]) ** 2, axis=1)
            for i in range(k):
                if i == labels[j]:
                    avgDWithin[j] = np.sum(distj[mbrs[:, i]]) / max(count[i] - 1, 1)
                else:
                    avgDBetween[j, i] = np.sum(distj[mbrs[:, i]]) / count[i]
        minavgDBetween = np.min(avgDBetween, axis=1)
        silh = (minavgDBetween - avgDWithin) / np.maximum(avgDWithin, minavgDBetween)
        return silh



# Example usage
if __name__ == "__main__":
    eyedat = [...]  # Your input data
    detector = ClusterFixationDetector()
    fixationstats = detector.detect_fixations(eyedat)
