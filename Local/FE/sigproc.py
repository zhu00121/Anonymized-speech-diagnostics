import os
import joblib
import pickle as pkl
import numpy as np
from srmrpy import srmr
import pandas as pd
from tqdm import tqdm
import librosa


def zeropad(x, fs=16000, length=10):
    """
    Take only a desired length of the signal.
    If not long enough, zeropad signal to a desired length.

    """
    xp = np.ones((length*fs,))*1e-15
    m = min(length*fs,x.shape[0])
    xp[:m,] = x[:m,]
    return xp


def calc_MFCC(x, fs, n_mfcc, window_length, hop_length, n_fft):

    """
    Calculate MFCC from a signal.
    """
    mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mfcc, \
            window_length = window_length, hop_length = hop_length, \
            n_fft = n_fft)
    
    return mfcc


def calc_melspec(x, fs, n_mfcc, window_length, hop_length, n_fft, require_log=True):

    """
    Calculate (log) mel-spectrogram from a signal.
    """
    mel_spec = librosa.feature.melspectrogram(y=x, sr=fs, n_mfcc=n_mfcc, \
            window_length = window_length, hop_length = hop_length, \
            n_fft = n_fft)

    if require_log:
        log_mel_spec = librosa.power_to_db(mel_spec)

    return mel_spec


def calc_mtr(x, fs, n_cochlear_filters, low_freq, min_cf, max_cf, fast=True, norm=True):

    """
    Calculate 3D modulation tensorgram representation (MTR). Each snapshot is a filterbank-applied modulation
    spectrum.
    """

    _, mtr = srmr(x, fs=fs, n_cochlear_filters=23, \
        low_freq=125, min_cf=2, max_cf=32, fast=fast, norm=norm)
    
    return mtr