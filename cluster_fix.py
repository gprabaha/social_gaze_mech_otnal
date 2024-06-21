#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:28:25 2024

@author: pg496
"""

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

def cluster_fix(eyedat, samprate=5/1000):
    if not eyedat:
        raise ValueError("No data file found")

    variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']
    fltord = 60
    lowpasfrq = 30
    nyqfrq = 1000 / 2
    flt = signal.firwin2(fltord, [0, lowpasfrq / nyqfrq, lowpasfrq / nyqfrq, 1], [1, 1, 0, 0])
    buffer = int(100 / (samprate * 1000))
    fixationstats = []

    for cndlop in range(len(eyedat)):
        if len(eyedat[cndlop][0]) > int(500 / (samprate * 1000)):
            x, y = preprocess_data(eyedat[cndlop], buffer, samprate, flt)
            vel, accel, angle, dist, rot = extract_parameters(x, y)
            points = normalize_parameters(dist, vel, accel, rot)

            T, meanvalues, stdvalues = global_clustering(points)
            fixationcluster, fixationcluster2 = find_fixation_clusters(meanvalues, stdvalues)
            T = classify_fixations(T, fixationcluster, fixationcluster2)

            fixationindexes, fixationtimes = behavioral_index(T, 1)
            fixationtimes = apply_duration_threshold(fixationtimes, 25)

            notfixations = local_reclustering(fixationtimes, points)
            fixationindexes = remove_not_fixations(fixationindexes, notfixations)
            saccadeindexes, saccadetimes = classify_saccades(fixationindexes, points)

            fixationtimes, saccadetimes = round_times(fixationtimes, saccadetimes, samprate)

            pointfix, pointsac, recalc_meanvalues, recalc_stdvalues = calculate_cluster_values(fixationtimes, saccadetimes, eyedat[cndlop], samprate)

            fixationstats.append({
                'fixationtimes': fixationtimes,
                'fixations': extract_fixations(fixationtimes, eyedat[cndlop]),
                'saccadetimes': saccadetimes,
                'FixationClusterValues': pointfix,
                'SaccadeClusterValues': pointsac,
                'MeanClusterValues': recalc_meanvalues,
                'STDClusterValues': recalc_stdvalues,
                'XY': np.array([eyedat[cndlop][0], eyedat[cndlop][1]]),
                'variables': variables
            })
        else:
            fixationstats.append({
                'fixationtimes': [],
                'fixations': [],
                'saccadetimes': [],
                'FixationClusterValues': [],
                'SaccadeClusterValues': [],
                'MeanClusterValues': [],
                'STDClusterValues': [],
                'XY': np.array([eyedat[cndlop][0], eyedat[cndlop][1]]),
                'variables': variables
            })

    return fixationstats


def preprocess_data(eyedat, buffer, samprate, flt):
    x = np.pad(eyedat[0], (buffer, buffer), 'reflect')
    y = np.pad(eyedat[1], (buffer, buffer), 'reflect')
    x = resample_data(x, samprate)
    y = resample_data(y, samprate)
    x = apply_filter(x, flt)
    y = apply_filter(y, flt)
    x = x[100:-100]
    y = y[100:-100]
    return x, y


def resample_data(data, samprate):
    t_old = np.linspace(0, len(data) - 1, len(data))
    t_new = np.linspace(0, len(data) - 1, int(len(data) * (samprate * 1000)))
    f = interp1d(t_old, data, kind='linear')
    return f(t_new)


def apply_filter(data, flt):
    return signal.filtfilt(flt, 1, data)


def extract_parameters(x, y):
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


def normalize_parameters(dist, vel, accel, rot):
    points = np.stack([dist, vel, accel, rot], axis=1)
    for ii in range(points.shape[1]):
        thresh = np.mean(points[:, ii]) + 3 * np.std(points[:, ii])
        points[points[:, ii] > thresh, ii] = thresh
        points[:, ii] = points[:, ii] - np.min(points[:, ii])
        points[:, ii] = points[:, ii] / np.max(points[:, ii])
    return points


def global_clustering(points):
    sil = np.zeros(5)
    for numclusts in range(2, 6):
        T = KMeans(n_clusters=numclusts, n_init=5).fit(points[::10, 1:4])
        silh = inter_vs_intra_dist(points[::10, 1:4], T.labels_)
        sil[numclusts - 2] = np.mean(silh)
    numclusters = np.argmax(sil) + 2
    T = KMeans(n_clusters=numclusters, n_init=5).fit(points)
    labels = T.labels_
    meanvalues = np.array([np.mean(points[labels == i], axis=0) for i in range(numclusters)])
    stdvalues = np.array([np.std(points[labels == i], axis=0) for i in range(numclusters)])
    return labels, meanvalues, stdvalues


def find_fixation_clusters(meanvalues, stdvalues):
    fixationcluster = np.argmin(np.sum(meanvalues[:, 1:3], axis=1))
    fixationcluster2 = np.where(meanvalues[:, 1] < meanvalues[fixationcluster, 1] + 3 * stdvalues[fixationcluster, 1])[0]
    fixationcluster2 = fixationcluster2[fixationcluster2 != fixationcluster]
    return fixationcluster, fixationcluster2


def classify_fixations(T, fixationcluster, fixationcluster2):
    T[T == fixationcluster] = 100
    for cluster in fixationcluster2:
        T[T == cluster] = 100
    T[T != 100] = 2
    T[T == 100] = 1
    return T


def behavioral_index(T, label):
    indexes = np.where(T == label)[0]
    return indexes, find_behavioral_times(indexes)


def find_behavioral_times(indexes):
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


def apply_duration_threshold(times, threshold):
    return times[:, np.diff(times, axis=0)[0] >= threshold]


def local_reclustering(fixationtimes, points):
    notfixations = []
    for fix in fixationtimes.T:
        altind = np.arange(fix[0] - 50, fix[1] + 50)
        altind = altind[(altind >= 0) & (altind < len(points))]
        POINTS = points[altind]
        sil = np.zeros(5)
        for numclusts in range(1, 6):
            T = KMeans(n_clusters=numclusts, n_init=5).fit(POINTS[::5])
            silh = inter_vs_intra_dist(POINTS[::5], T.labels_)
            sil[numclusts - 1] = np.mean(silh)
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
    return np.array(notfixations)


def remove_not_fixations(fixationindexes, notfixations):
    fixationindexes = np.setdiff1d(fixationindexes, notfixations)
    return fixationindexes


def classify_saccades(fixationindexes, points):
    saccadeindexes = np.setdiff1d(np.arange(len(points)), fixationindexes)
    saccadetimes = find_behavioral_times(saccadeindexes)
    return saccadeindexes, saccadetimes


def round_times(fixationtimes, saccadetimes, samprate):
    round5 = np.mod(fixationtimes, samprate * 1000)
    round5[0, round5[0] > 0] = samprate * 1000 - round5[0, round5[0] > 0]
    round5[1] = -round5[1]
    fixationtimes = np.round((fixationtimes + round5) / (samprate * 1000)).astype(int)
    fixationtimes[fixationtimes < 1] = 1

    round5 = np.mod(saccadetimes, samprate * 1000)
    round5[0] = -round5[0]
    round5[1, round5[1] > 0] = samprate * 1000 - round5[1, round5[1] > 0]
    saccadetimes = np.round((saccadetimes + round5) / (samprate * 1000)).astype(int)
    saccadetimes[saccadetimes < 1] = 1

    return fixationtimes, saccadetimes


def calculate_cluster_values(fixationtimes, saccadetimes, eyedat, samprate):
    x = eyedat[0]
    y = eyedat[1]
    pointfix = [extract_variables(x[fix[0]:fix[1]], y[fix[0]:fix[1]], samprate) for fix in fixationtimes.T]
    pointsac = [extract_variables(x[sac[0]:sac[1]], y[sac[0]:sac[1]], samprate) for sac in saccadetimes.T]
    recalc_meanvalues = [np.nanmean(pointfix, axis=0), np.nanmean(pointsac, axis=0)]
    recalc_stdvalues = [np.nanstd(pointfix, axis=0), np.nanstd(pointsac, axis=0)]
    return pointfix, pointsac, recalc_meanvalues, recalc_stdvalues


def extract_fixations(fixationtimes, eyedat):
    x = eyedat[0]
    y = eyedat[1]
    fixations = [np.mean([x[fix[0]:fix[1]], y[fix[0]:fix[1]]], axis=1) for fix in fixationtimes.T]
    return np.array(fixations)


def extract_variables(xss, yss, samprate):
    if len(xss) < 3:
        return np.full(6, np.nan)
    vel = np.sqrt(np.diff(xss) ** 2 + np.diff(yss) ** 2) / samprate
    angle = np.degrees(np.arctan2(np.diff(yss), np.diff(xss)))
    accel = np.abs(np.diff(vel)) / samprate
    dist = [np.sqrt((xss[a] - xss[a + 2]) ** 2 + (yss[a] - yss[a + 2]) ** 2) for a in range(len(xss) - 2)]
    rot = [np.abs(angle[a] - angle[a + 1]) for a in range(len(xss) - 2)]
    rot = [r if r <= 180 else 360 - r for r in rot]
    return [np.max(vel), np.max(accel), np.mean(dist), np.mean(vel), np.abs(np.mean(angle)), np.mean(rot)]


def inter_vs_intra_dist(X, labels):
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
