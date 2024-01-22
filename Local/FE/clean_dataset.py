"""
Three datasets are included in this study:
1. ComParE 2022 CSS; 2. DiCOVA2; 3. Task-2 set from Cambridge database.

As they have different ways to store the sound files and labels, 
we here write a script to create one .csv file for each dataset which includes three columns:
audio_path|label|split

"""

import pandas as pd
import numpy as np
import os, glob, argparse
import functools

def clean_CSS(**kwargs):

    # load train&devel&test labels (.csv) 
    df_label_train = pd.read_csv(kwargs['label_train_path'])
    df_label_valid = pd.read_csv(kwargs['label_valid_path'])
    df_label_test = pd.read_csv(kwargs['label_test_path'])

    # use sample id to locate audio path
    filename_train = df_label_train['filename'].to_list()
    filename_valid = df_label_valid['filename'].to_list()
    filename_test = df_label_test['filename'].to_list()
    
    def reconstruct_ad_name(filename:str,split:str,ad_folder:str):
        _, tail = filename.split('_')
        ad_name = '%s_'%(split) + tail.rjust(3,'0') + '.wav'
        ad_path = os.path.join(ad_folder,ad_name)
        return ad_path

    audio_path_train = map(functools.partial(reconstruct_ad_name,split='train',ad_folder=kwargs['ad_folder']),filename_train)
    audio_path_valid = map(functools.partial(reconstruct_ad_name,split='dev',ad_folder=kwargs['ad_folder']),filename_valid)
    audio_path_test = map(functools.partial(reconstruct_ad_name,split='test',ad_folder=kwargs['ad_folder']),filename_test)
    
    # add one column to indicate split (train/valid/test)
    df_train_new = pd.DataFrame({'audio_path':audio_path_train, \
                                'label':df_label_train['label'], \
                                'split':'train'})
    df_valid_new = pd.DataFrame({'audio_path':audio_path_valid, \
                                'label':df_label_valid['label'], \
                                'split':'valid'})
    df_test_new = pd.DataFrame({'audio_path':audio_path_test, \
                                'label':df_label_test['label'], \
                                'split':'test'})
    
    # concatenate three tables into one
    df_all = pd.concat([df_train_new,df_valid_new,df_test_new],axis=0, ignore_index=True)
    codes = {'negative':0, 'positive':1}
    df_all['label'] = df_all['label'].map(codes)

    # test if path is valid
    print('Test if all audio path exist...')
    assert any([os.path.exists(i) for i in df_all['audio_path'].to_list()]), "audio path does not exist"

    # save new dataframe as .csv file
    df_all.to_csv(os.path.join(os.path.split(kwargs['label_train_path'])[0],'metadata_v2.csv'))

    return df_all


def clean_DiCOVA2(**kwargs):

    # load metadata (.csv) and pre-defined split (.csv)
    df_md = pd.read_csv(kwargs['label_path'],delimiter=r'\s+')
    filename_all = df_md['SUB_ID'].to_list()

    # substitute sample_id with absolute file path
    def reconstruct_ad_name(filename:str,ad_folder:str):
        ad_name = filename + '.flac'
        ad_path = os.path.join(ad_folder,ad_name)
        return ad_path

    ad_path = map(functools.partial(reconstruct_ad_name,ad_folder=kwargs['ad_folder']),filename_all)
    
    # get split assigned to each sample. we here load a pre-defined split.
    pd_train = pd.read_csv(kwargs['pd_train'],header=None)
    pd_train = pd_train.iloc[:,0].to_list()
    pd_test = pd.read_csv(kwargs['pd_test'],header=None)
    pd_test = pd_test.iloc[:,0].to_list()
    split_col = []
    for sample in filename_all:
        if sample in pd_train:
            split_col.append('train')
        elif sample in pd_test:
            split_col.append('test')

    # add one column to indicate split (train/valid/test)
    codes = {'n':0, 'p':1}
    df_md['COVID_STATUS'] = df_md['COVID_STATUS'].map(codes)

    df_md_new = pd.DataFrame({
        'audio_path':ad_path,
        'label':df_md['COVID_STATUS'],
        'split':split_col
    })

    # test if audio path is valid
    print('Test if all audio path exist...')
    assert any([os.path.exists(i) for i in df_md_new['audio_path'].to_list()]), "audio path does not exist"

    # save new dataframe as .csv file
    df_md_new.to_csv(os.path.join(os.path.split(kwargs['label_path'])[0],'metadata_v2.csv'))

    return df_md_new


def clean_Cambridge(**kwargs):

    # load labels (.csv)
    df_md = pd.read_csv(kwargs['label_path'])
    voice_path = df_md['voice_path'].to_list()
    voice_path = [i.replace("\\","/") for i in voice_path]
    df_md['voice_path'] = [os.path.join(kwargs['ad_folder'],i) for i in voice_path]

    # modify columns
    df_md_new = pd.DataFrame({
        'audio_path':df_md['voice_path'],
        'label':df_md['label'],
        'split':df_md['fold']
    })
    
    # test if path is valid
    print('Test if all audio path exist...')
    assert any([os.path.exists(i) for i in df_md_new['audio_path'].to_list()]), "audio path does not exist"

    # save new dataframe as .csv file
    df_md_new.to_csv(os.path.join(os.path.split(kwargs['label_path'])[0],'metadata_v2.csv'))

    return df_md_new


def open_kwargs(filepath):
    with open(filepath, "r") as file:
        kwargs = eval(file.read())
    return kwargs


def save_kwargs(filepath,kwargs):
    with open(filepath, "w") as file:
        file.write(str(kwargs))
    return 0


def main():

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)
    
    def file_path(string):
        if os.path.exists(string):
            return string
        else:
            raise FileNotFoundError('No folder exists at the location specified')

    base_parser = argparse.ArgumentParser(prog='Create metadata files')
    base_parser.add_argument('--ad_folder', type=dir_path, required=True)
    subparsers = base_parser.add_subparsers(dest='dataset', help='Choose datasets')
    subparsers.required = True

    CSS_parser = subparsers.add_parser('CSS')
    CSS_parser.add_argument('--label_train_path', type=file_path, required=True)
    CSS_parser.add_argument('--label_valid_path', type=file_path, required=True)
    CSS_parser.add_argument('--label_test_path', type=file_path, required=True)

    DiCOVA2_parser = subparsers.add_parser('DiCOVA2')
    DiCOVA2_parser.add_argument('--label_path', type=file_path, required=True)
    DiCOVA2_parser.add_argument('--pd_train', type=file_path, required=True)
    DiCOVA2_parser.add_argument('--pd_test', type=file_path, required=True)

    Cambridge_parser = subparsers.add_parser('Cambridge')
    Cambridge_parser.add_argument('--label_path', type=file_path, required=True)

    args = base_parser.parse_args()

    if args.dataset == 'CSS':
        print('Start processing CSS...')
        kwargs = {
            'label_train_path':args.label_train_path,
            'label_valid_path':args.label_valid_path,
            'label_test_path':args.label_test_path,
            'ad_folder':args.ad_folder
        }
        clean_CSS(**kwargs)
        print('File saved to: '+os.path.dirname(kwargs['ad_folder']))

    elif args.dataset == 'DiCOVA2':
        print('Start processing DiCOVA2...')
        kwargs = {
            'label_path':args.label_path,
            'pd_train':args.pd_train,
            'pd_test':args.pd_test,
            'ad_folder':args.ad_folder
        }
        clean_DiCOVA2(**kwargs)
        print('File saved to: '+os.path.dirname(kwargs['label_path']))
    
    elif args.dataset == 'Cambridge':
        print('Start processing Cambridge Task-2...')
        kwargs = {
            'label_path':args.label_path,
            'ad_folder':args.ad_folder
        }
        clean_Cambridge(**kwargs)
        print('File saved to: '+os.path.dirname(kwargs['ad_folder']))


if __name__ == '__main__':

    main()