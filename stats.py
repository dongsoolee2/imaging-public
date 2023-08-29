"""
Statistics functions to analyze retinal physiology data
"""

import numpy as np
import itertools
import scipy.stats
import scipy.ndimage
import copy

__all__ = ['mean', 'corr', 'corr_pair', 'corr_mean', 'pearsonr_mask',
        'normalize', 'zero_one', 'smooth', 'get_f', 'df_over_f', 'get_var', 
        'var_normalize']

def mean(arr, axis=0, use_half_trial=0):
    D = arr.ndim
    if D == 4:
        _, _, N, _ = arr.shape
        if use_half_trial:
            N = int(N/2)
            arr = arr[:, :, :N, :]
    elif D == 3:
        _, N, _ = arr.shape
        if use_half_trial:
            N = int(N/2)
            arr = arr[:, :N, :]
    elif D == 2:
        N, _ = arr.shape
        if use_half_trial:
            N = int(N/2)
            arr = arr[:N, :]
    return np.nanmean(arr, axis=axis)


def corr(arr, mode='pair', use_half_trial=0):
    D = arr.ndim
    if D == 4:
        D1, D2, N, T = arr.shape
        if use_half_trial:
            N = int(N/2)
            arr = arr[:, :, :N, :]
        if mode == 'pair':
            C = int(N * (N - 1) / 2)
            result = np.zeros((D1, D2, C, 2))
            for d1 in range(D1):
                for d2 in range(D2):
                    result[d1, d2, :, :] = corr_pair(arr[d1, d2, :, :])
            return result
        elif mode == 'mean':
            result = np.zeros((D1, D2, N, 2))
            for d1 in range(D1):
                for d2 in range(D2):
                    result[d1, d2, :, :] = corr_mean(arr[d1, d2, :, :])
            return result
    elif D == 3:
        D1, N, T = arr.shape
        if use_half_trial:
            N = int(N/2)
            arr = arr[:, :N, :]
        if mode == 'pair':
            C = int(N * (N - 1) / 2)
            result = np.zeros((D1, C, 2))
            for d1 in range(D1):
                result[d1, :, :] = corr_pair(arr[d1, :, :])
            return result
        elif mode == 'mean':
            result = np.zeros((D1, N, 2))
            for d1 in range(D1):
                result[d1, :, :] = corr_mean(arr[d1, :, :])
            return result
    elif D == 2:
        N, T = arr.shape
        if use_half_trial:
            N = int(N/2)
            arr = arr[:N, :]
        if mode == 'pair':
            return corr_pair(arr)
        elif mode == 'mean':
            return corr_mean(arr)

def corr_pair(arr):
    N, T = arr.shape
    C = int(N * (N - 1) / 2)
    result = np.zeros((C, 2))
    for i, (a, b) in enumerate(itertools.combinations(range(N), 2)):
        cc, p = pearsonr_mask(arr[a, :], arr[b, :])
        result[i, 0] = cc
        result[i, 1] = p
    return result

def corr_mean(arr):
    N, T = arr.shape
    arr_mean = np.nanmean(arr, axis=0)
    result = np.zeros((N, 2))
    for i in range(N):
        cc, p = pearsonr_mask(arr[i, :], arr_mean)
        result[i, 0] = cc
        result[i, 1] = p
    return result

def pearsonr_mask(a, b):
    mask = np.logical_or(np.isnan(a), np.isnan(b))
    return scipy.stats.pearsonr(a[~mask], b[~mask])

def normalize(arr):
    temp = arr.copy()
    temp -= np.mean(temp)
    temp /= np.std(temp)
    return temp

def zero_one(arr):
    temp = arr.copy()
    temp -= np.min(temp)
    temp /= np.max(temp)
    return temp

def smooth(arr, w):
    return scipy.ndimage.gaussian_filter1d(arr, w)

def get_f(arr, w=1200):
    win1_2 = int(w/2)
    T = arr.shape[0]
    F = np.zeros(T)
    for i in range(T):
        F[i] = np.nanmean(arr[np.max([i - win1_2, 0]):np.min([i + win1_2, T])])
    return F

def df_over_f(arr, w=1200):
    F = get_f(arr, w)
    return (arr - F)/abs(F + 1)

def get_var(arr, w=1200):
    win1_2 = int(w/2)
    T = arr.shape[0]
    V = np.zeros(T)
    for i in range(T):
        V[i] = np.var(arr[np.max([i - win1_2, 0]):np.min([i + win1_2, T])])
    return V

def var_normalize(arr, w=1200):
    temp = arr.copy()
    V = get_var(temp, w)
    return temp/np.sqrt(V)
