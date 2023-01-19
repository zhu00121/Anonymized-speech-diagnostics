import os,sys
sys.path.append("./Local/")
import joblib
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
# below are the two signal processing libraries used
import librosa
from srmrpy import srmr
import sigproc as sp
from Utils import util
import scipy.io.wavfile


def FE_from_single_ad(audio_path:str, fs:int, feature_type:str, save_path:str, **kwargs):

    """
    Extract features from a single audio file.
    """
    # Read audio file from specified path
    ad_og = sp.read_audio(audio_path=audio_path, fs=fs)
    if kwargs.get('zeropad_len'):
        ad = sp.zeropad(x=ad_og,fs=fs,length=kwargs['zeropad_len'])

    assert feature_type in ['logmelspec','openSMILE','msr','mtr', 'mtr_v2', 'mtr_v3'], "Incorrect feature type"

    if feature_type == 'logmelspec':
        ot = sp.calc_melspec(x=ad, fs=fs, n_mels=kwargs['n_mels'], \
                            win_length=kwargs['win_len'], hop_length=kwargs['hop_len'], \
                            n_fft=kwargs['n_fft'], require_log=True)

        ot_d = sp.calc_deltas(x=ot)
        ot = np.concatenate((ot,ot_d),axis=0)

    elif feature_type == 'openSMILE':
        ot = sp.calc_openSMILE(ad_og)

    elif feature_type == 'msr' or 'mtr' in feature_type:
        ot = sp.calc_mtr(x=ad, fs=fs, \
            n_cochlear_filters=kwargs['n_cochlear_filters'], low_freq=kwargs['low_freq'],\
            min_cf=kwargs['min_cf'], max_cf=kwargs['max_cf'], \
            require_ave=kwargs['require_ave'])

    assert np.isfinite(ot).all(), "NaN in extracted features!"

    # Save result as a pkl file
    util.save_as_pkl(save_path,ot)

    return ot


def _get_alternative_ad(fs:int):

    alter_path = './noise.wav'
    if not os.path.exists(alter_path):
        toy = np.random.randn((fs*3))
        scipy.io.wavfile.write(alter_path, fs, np.float32(toy))
    return alter_path
    

def FE_from_dataset(metadata_path:str, feature_dir:str, fs:int, feature_type:str, anonymize:str, **kwargs):

    """
    *Ensure that 'clean_dataset.py' runs before feature extraction.
    Extract features for the whole dataset:
    1. Access the metadata file which contains path to every speech recording;
    2. Extract features recursively from each recording;
    3. Store the extracted features separately for each sample;
    4. Update the metadata file with feature path.
    """
    assert os.path.exists(metadata_path), "Required metadata file is not in this location"

    if not os.path.exists(feature_dir):
        print('Input feature directory does not exist, creating one now...')
        os.makedirs(feature_dir)
        print('Input feature directory had been created successfully')

    assert feature_type in ['logmelspec','openSMILE','msr', 'mtr','mtr_v2','mtr_v3'], "Incorrect feature type"
    assert anonymize in ['og','mcadams','ss'], "Input anonymize approach is not supported"

    df_md = pd.read_csv(metadata_path) # load metadata
    assert 'audio_path' in df_md.columns.values, "Required info missing in metadata"

    all_audio_path = df_md['audio_path'].tolist()
    all_feature_path = [] # to store feature path (./XXX.pkl)

    print('Start feature extraction...')
    # create uniq id for each recording in case of duplicated names (happens with Cambridge set)
    uniq = 0
    error_list = []
    for ad_path in tqdm(all_audio_path):
        # get sample id
        _, tail = os.path.split(ad_path)
        sample_id = util.remove_suffix(tail,'.wav')
        sample_id = util.remove_suffix(sample_id,'.flac')
        feature_path = os.path.join(feature_dir,'%s_%d.pkl'%(sample_id,uniq)) # feature path
        all_feature_path.append(feature_path)

        if os.path.exists(ad_path):
            try:
                ot = FE_from_single_ad(ad_path, fs, feature_type, feature_path, **kwargs)
            except:
                print('Error in '+sample_id+'. Extracting features from back-up audio.')
                error_list.append(sample_id)
                ot = FE_from_single_ad(_get_alternative_ad(fs), fs, feature_type, feature_path, **kwargs)
        
        elif not os.path.exists(ad_path):
            print(sample_id+' does not exist. Extracting features from back-up audio.')
            error_list.append(sample_id)
            ot = FE_from_single_ad(_get_alternative_ad(fs), fs, feature_type, feature_path, **kwargs)

        uniq += 1
    print('------')
    print('Feature extraction completed.')
    print(error_list)

    # Save feature path
    df_md_new = df_md
    df_md_new['feature_path_%s'%(feature_type)] = all_feature_path
    metadata_path_new = os.path.join(os.path.dirname(metadata_path),'metadata_%s.csv'%(anonymize))
    df_md_new.to_csv(metadata_path_new)

    return df_md_new


if __name__ == '__main__':

    # test feature extraction from the whole dataset
    # kwargs = {'n_mels':64, 'win_len':256, 'hop_len':64, 'n_fft':512}
    # kwargs = {'n_cochlear_filters':23,'low_freq':0, 'min_cf':2,'max_cf':32,'require_ave':True}
    # kwargs = util.load_json('./Config/feat_config/logmelspec_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_og.csv',
    #     feature_dir='./Features/DiCOVA2/original/logmelspec',
    #     fs=16000,
    #     feature_type='logmelspec',
    #     anonymize='og',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/logmelspec_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/CSS/label/metadata_ss.csv',
    #     feature_dir='./Features/CSS/ss/logmelspec',
    #     fs=16000,
    #     feature_type='logmelspec', 
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/logmelspec_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_ss.csv',
    #     feature_dir='./Features/DiCOVA2/ss/logmelspec',
    #     fs=16000,
    #     feature_type='logmelspec',
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/logmelspec_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_ss.csv',
    #     feature_dir='./Features/Cambridge/ss/logmelspec',
    #     fs=16000,
    #     feature_type='logmelspec',
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/msr_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/CSS/label/metadata_ss.csv',
    #     feature_dir='./Features/CSS/ss/msr',
    #     fs=16000,
    #     feature_type='msr',
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/msr_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_ss.csv',
    #     feature_dir='./Features/DiCOVA2/ss/msr',
    #     fs=16000,
    #     feature_type='msr',
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/msr_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_ss.csv',
    #     feature_dir='./Features/Cambridge/ss/msr',
    #     fs=16000,
    #     feature_type='msr',
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/mtr_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/CSS/label/metadata_ss.csv',
    #     feature_dir='./Features/CSS/ss/mtr',
    #     fs=16000,
    #     feature_type='mtr',
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/mtr_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_ss.csv',
    #     feature_dir='./Features/DiCOVA2/ss/mtr',
    #     fs=16000,
    #     feature_type='mtr',
    #     anonymize='ss',
    #     **kwargs
    # )

    # kwargs = util.load_json('./Config/feat_config/mtr_config')
    # ot = FE_from_dataset(
    #     metadata_path='/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_ss.csv',
    #     feature_dir='./Features/Cambridge/ss/mtr',
    #     fs=16000,
    #     feature_type='mtr',
    #     anonymize='ss',
    #     **kwargs
    # )

    ot = FE_from_dataset(
        metadata_path='/mnt/d/projects/COVID-datasets/CSS/label/metadata_ss.csv',
        feature_dir='./Features/CSS/ss/openSMILE',
        fs=16000,
        feature_type='openSMILE',
        anonymize='ss'
    )


    ot = FE_from_dataset(
        metadata_path='/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_ss.csv',
        feature_dir='./Features/DiCOVA2/ss/openSMILE',
        fs=16000,
        feature_type='openSMILE',
        anonymize='ss'
    )


    ot = FE_from_dataset(
        metadata_path='/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_ss.csv',
        feature_dir='./Features/Cambridge/ss/openSMILE',
        fs=16000,
        feature_type='openSMILE',
        anonymize='ss'
    )