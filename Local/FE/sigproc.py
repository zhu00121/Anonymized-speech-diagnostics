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
    ad = ad/np.max(ad) # amplitude scaling
    plt.plot(ad)
    plt.show()
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
    """
    _, mtr = srmr(x, fs=fs, n_cochlear_filters=n_cochlear_filters, \
        low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)
    
    if require_ave:
        mtr = np.mean(mtr,axis=0).squeeze() # average over the time axis

    return mtr


def calc_openSMILE(audio_path):

    """
    Calculate openSMILE features - ComParE_2016 version which has 6,000ish features.
    """
    assert os.path.exists(audio_path), "Audio path does not exist!"

    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals)
    y = smile.process_file(audio_path)

    return y


def save_as_pkl(save_path:str,variable):

    """
    Save result as a pickle file.
    """
    with open(save_path, "wb") as outfile:
        pkl.dump(variable, outfile, pkl.HIGHEST_PROTOCOL)
        
    return 0


def load_pkl(save_path:str):

    """
    Load saved pickle file.
    """
    assert os.path.exists(save_path), "pkl file does not exist!"
    with open(save_path, "rb") as infile:
        saved_fea = pkl.load(infile)

    return saved_fea