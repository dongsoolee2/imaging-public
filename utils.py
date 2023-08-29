"""
General functions to help analyze retinal physiology data
"""

import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

__all__ = ['butter_lf', 'butter_hf', 'load_stimulus_wn', 'rescale'
        'color_index', 'set_plot_params']

def butter_lf(data, Wn_l=6.0, fs=30.0, order=3):
    b_l, a_l = scipy.signal.butter(N=order, Wn=Wn_l, btype='lowpass', fs=fs)
    return scipy.signal.filtfilt(b_l, a_l, data)

def butter_hf(data, Wn_h=0.04, fs=30.0, order=3):
    b_h, a_h = scipy.signal.butter(N=order, Wn=Wn_h, btype='highpass', fs=fs)
    return scipy.signal.filtfilt(b_h, a_h, data)

def load_stimulus_wn(f):
    mat = scipy.io.loadmat(f)
    stim = mat[sorted(mat.keys())[-1]].swapaxes(2, 1)
    T = stim.shape[1]
    xx = stim.shape[2]
    x = int(np.sqrt(xx))
    temp = np.double(stim.reshape(-1, T, x, x)[1])
    return rescale(temp)

def rescale(arr):
    arr /= 255
    arr -= 0.5
    arr *= 2
    return arr

def color_index():
    color_code = ['#E6194B', '#3CB44B', '#FFE119', '#0082C8',
            '#F58231', '#911EB4', '#46F0F0', '#F032E6',
            '#D2F53C', '#FABEBE', '#008080', '#E6BEFF',
            '#AA6E28', '#800000', '#AAFFC3', '#808000',
            '#FFD8B1', '#000080', '#808080', '#FFFAC8']
    return color_code

def set_plot_params():
    plt.rcParams['figure.figsize'] = [4.0, 2.0]
  #  plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 2
    plt.rcParams['axes.linewidth'] = 0.7
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3