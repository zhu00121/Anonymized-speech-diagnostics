import os
import joblib
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
# below are the two signal processing libraries used
import librosa
from srmrpy import srmr
import opensmile
import matplotlib.pyplot as plt


def read_audio(audio_path:str, fs:int):

    """
    Read in an audio file from the specified path.
    """
    assert os.path.exists(audio_path), "Audio path does not exist!"
    ad,_ = librosa.load(audio_path,sr=fs) # sampling rate variable discarded
    ad = ad/(np.max(ad)+np.finfo(np.float32).eps) # amplitude scaling
    assert np.isfinite(ad).all(), "NaN in audio array"

    return ad


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


def calc_melspec(x, fs, n_mels, win_length, hop_length, n_fft, require_log=True):

    """
    Calculate (log) mel-spectrogram from a signal.
    Returns: np.ndarray [shape=(…, n_mels, t)]
    """
    mel_spec = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_mels, \
            win_length = win_length, hop_length = hop_length, \
            n_fft = n_fft)

    if require_log:
        log_mel_spec = librosa.power_to_db(mel_spec)

    return mel_spec


def calc_deltas(x, width=5, order='both'):

    """
    Calculate 1st and/or 2nd-order deltas based on input features.
    """
    delta_1 = librosa.feature.delta(x,width=width,order=1)
    delta_2 = librosa.feature.delta(x,width=width,order=2)
    delta_ot = np.concatenate((delta_1,delta_2),axis=0)

    return delta_ot


def calc_mtr(x, fs, n_cochlear_filters, low_freq, min_cf, max_cf, fast=True, norm=True, require_ave=True):

    """
    Calculate 3D modulation tensorgram representation (MTR). Each snapshot is a filterbank-applied modulation
    spectrum.
    Returns: np.ndarray [shape=(23, 8, t)]
    """

    _, mtr = srmr(x, fs=fs, n_cochlear_filters=n_cochlear_filters, \
        low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)
    
    assert mtr.ndim == 3

    if require_ave:
        mtr = np.mean(mtr,axis=2).squeeze() # average over the time axis
        mtr = mtr.flatten()

    return mtr


def calc_openSMILE(ad):

    """
    Calculate openSMILE features - ComParE_2016 version which has 6,000ish features.
    """

    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals)
    df_y = smile.process_signal(ad,16000) # y is a DataFrame
    y = df_y.to_numpy()

    return y
