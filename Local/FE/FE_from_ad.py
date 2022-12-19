import os
import joblib
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
# below are the two signal processing libraries used
import librosa
from srmrpy import srmr
import sigproc as sp


def FE_from_single_ad(audio_path:str, fs:int, feature_type:str, save_path:str, **kwargs):

    """
    Extract features from a single audio file.
    """
    # Read audio file from specified path
    ad = sp.read_audio(audio_path=audio_path, fs=fs)
    if kwargs['require_zeropad']:
        ad = sp.zeropad(x=ad,fs=fs,length=kwargs['desired_len'])

    assert feature_type in ['logmelspec','openSMILE','msr'], "Incorrect feature type"

    if feature_type == 'logmelspec':
        ot = sp.calc_melspec(x=ad, fs=fs, n_mels=kwargs['n_mels'], \
                            win_length=kwargs['win_len'], hop_length=kwargs['hop_len'], \
                            n_fft=kwargs['n_fft'], require_log=True)

        ot_d = sp.calc_deltas(x=ot)
        ot = np.concatenate((ot,ot_d),axis=0)

    elif feature_type == 'openSMILE':
        ot = sp.calc_openSMILE(audio_path=audio_path)

    elif feature_type == 'msr':
        ot = sp.calc_mtr(x=ad, fs=fs, n_cochlear_filters=kwargs['n_cochlear_filters'], \
                        low_freq=kwargs['low_freq'], min_cf=kwargs['min_cf'], max_cf=kwargs['max_cf'], \
                        require_ave=kwargs['require_ave'])

    assert np.isfinite(ot).all(), "NaN in extracted features!"

    # Save result as a pkl file
    sp.save_as_pkl(save_path,ot)

    return ot


def FE_from_dataset(metadata_path:str, feature_dir:str, fs:int, feature_type:str, **kwargs):

    """
    *Ensure that 'clean_dataset.py' runs before feature extraction.
    Extract features for the whole dataset:
    1. Access the metadata file which contains path to every speech recording;
    2. Extract features recursively from each recording;
    3. Store the extracted features separately for each sample;
    4. Update the metadata file with feature path.
    """
    assert os.path.exists(metadata_path), "Required metadata file is not in this location"

    df_md = pd.read_csv(metadata_path) # load metadata
    all_audio_path = df_md['audio_path'].tolist()
    all_feature_path = [] # to store feature path (./XXX.pkl)

    print('Start feature extraction...')
    for idx, ad_path in tqdm(all_audio_path):
        # get sample id
        _, tail = os.path.split(ad_path)
        sample_id = tail.removesuffix('.wav')
        sample_id = sample_id.removesuffix('.flac')
        feature_path = os.path.join(feature_dir,'%s.pkl'%(sample_id)) # feature path
        all_feature_path.append(feature_path)
        ot = FE_from_single_ad(ad_path, fs, feature_type, feature_path, kwargs)
    print('------')
    print('Feature extraction completed.')

    # Save feature path
    df_md_new = df_md
    df_md_new['Feature_path'] = all_feature_path
    metadata_path_new = os.path.join(os.path.dirname(metadata_path),'metadata_feature.csv')
    df_md_new.to_csv(metadata_path_new)

    return df_md_new


if __name__ == '__main__':
    
    # test logmelspec 
    kwargs = {'require_zeropad':False,'n_mels':64, 'win_len':256, 'hop_len':64, 'n_fft':512}

    ot = FE_from_single_ad('/mnt/d/projects/COVID-datasets/CSS/audio/dev_001.wav',\
        fs=16000, feature_type='logmelspec', \
        save_path='/mnt/d/projects/Anonymized-speech-diagnostics/Features/CSS/original/toy.pkl',
        **kwargs)
