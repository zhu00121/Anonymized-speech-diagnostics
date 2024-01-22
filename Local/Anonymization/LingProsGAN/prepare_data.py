

"""
Aggregate speech recordings from multi sources 
and prepare the manifest file. Assuming this is called before
loading into pytorch dataloaders.
"""

import os, sys, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import random
import pandas as pd
import json
from path import Path
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from spk_emd import SpkEncoder

logger = logging.getLogger(__name__)

def prepare_cambridge(PATH_METADATA_FOLDER):
    """
    Generate a .csv file containing files paths, labels, and splits.
    Return a dataframe of the .csv file.
    """
    df = pd.read_csv(os.path.join(PATH_METADATA_FOLDER,'cambridge_metadata.csv'),sep=',')
    df['voice_path'] = df['voice_path'].str.replace('\\','/')
    df = df.drop(columns=['cough_path','breath_path','Index','categs']) # drop redundant columns for concat with other dfs.
    df.columns = ['voice-path-new', 'label', 'Uid', 'split']
    df['split'] = df['split'].map({'train':0, 'validation':1, 'test':2})
    df['voice-path-new'] = 'exp/covid_data/wav/Cambridge/audio/' + df['voice-path-new']
    return df

def prepare_dicova(PATH_METADATA_FOLDER):
    df_og = pd.read_csv(os.path.join(PATH_METADATA_FOLDER,'dicova2_metadata.csv'),sep='\s+') # reads the original metadata file
    df_og.columns = ['Uid', 'label', 'Sex']
    df_og['label'] = df_og['label'].map({'n':0, 'p':1})
    df_og['voice-path-new'] = None
    df_og['voice-path-new'] = 'exp/covid_data/wav/DiCOVA2/' + df_og['Uid'] + '.wav'
    # With dicova, we need to load the splits separately from the text files.
    train_split = pd.read_csv(os.path.join(PATH_METADATA_FOLDER,'dicova2_train.csv'),sep='\s+',header=None)
    test_split = pd.read_csv(os.path.join(PATH_METADATA_FOLDER,'dicova2_val.csv'),sep='\s+',header=None)
    df_og['split'] = 0
    for i in test_split.loc[:,0].to_list():
        df_og.loc[df_og['Uid'] == i, 'split'] = 2
    for i in random.sample(train_split.loc[:,0].to_list(),k=150):
        df_og.loc[df_og['Uid'] == i, 'split'] = 1
    df_og = df_og.drop(columns=['Sex']) # drop redundant columns for concat with other dfs.
    return df_og

def prepare_css(PATH_METADATA_FOLDER):
    df_train = pd.read_csv(os.path.join(PATH_METADATA_FOLDER,'css_train.csv'),sep=',')
    df_train['split'] = 0
    df_train['Uid'] = None
    df_train['Uid'] = 'train_' + df_train['filename'].str.split('_').str[1].str.rjust(3,"0")
    df_dev = pd.read_csv(os.path.join(PATH_METADATA_FOLDER,'css_dev.csv'),sep=',')
    df_dev['split'] = 1
    df_dev['Uid'] = None
    df_dev['Uid'] = 'dev_' + df_dev['filename'].str.split('_').str[1].str.rjust(3,"0")
    df_test = pd.read_csv(os.path.join(PATH_METADATA_FOLDER,'css_test.csv'),sep=',')
    df_test['split'] = 2
    df_test['Uid'] = None
    df_test['Uid'] = 'test_' + df_test['filename'].str.split('_').str[1].str.rjust(3,"0")
    df = pd.concat([df_train,df_dev,df_test], ignore_index=True)
    df = df.drop(columns=['filename'])
    df['voice-path-new'] = 'exp/covid_data/wav/CSS/' + df['Uid'] + '.wav'
    df['label'] = df['label'].map({'positive':1, 'negative':0})
    return df

def aggregate_covid_data_for_GAN(
        PATH_METADATA:str,
        metadata_filename:str,
        manifest_train_path,
        manifest_valid_path,
        manifest_test_path
        ):
    """_summary_

    Args:
        PATH_METADATA (str): path to the metadata folder
        metadata_filename (str): name of the metadata file
        manifest_train_path (_type_): path to the manifest file for training
        manifest_valid_path (_type_): path to the manifest file for validation
        manifest_test_path (_type_): path to the manifest file for testing
    """
    
    # Check if this phase is already done (if so, skip it)
    if skip(manifest_train_path, manifest_valid_path, manifest_test_path):
        logger.info("Manifest files preparation completed in previous run, skipping.")
        return

    # Prepare metadata files and aggregate into one file
    if not os.path.exists(os.path.join(PATH_METADATA,metadata_filename)):
        logger.info('metadata file does not exit. start creating one.')
        md_cam = prepare_cambridge(PATH_METADATA)
        md_css = prepare_css(PATH_METADATA)
        md_dic = prepare_dicova(PATH_METADATA)
        md_all = pd.concat([md_cam, md_css, md_dic], ignore_index=True)
        md_all.to_csv(os.path.join(PATH_METADATA,metadata_filename),index=False,header=True,sep=';')

    if not os.path.exists('exp/covid_data/spk_emd'):
        logger.info("Extracting speaker embeddings...")
        enc = SpkEncoder(emd_save_dir='exp/covid_data/spk_emd')
        enc.extract_from_group(os.path.join(PATH_METADATA,metadata_filename), debug_mode=False)

    # List files and create manifest from list
    logger.info(
        f"Creating {manifest_train_path}, {manifest_valid_path}, and {manifest_test_path}"
    )
    
    # Creating json files for train, valid, and test all at once
    create_json(
        os.path.join(PATH_METADATA,metadata_filename),
        [manifest_train_path,manifest_valid_path,manifest_test_path]
                )


def create_json(metadata_path:str, manifest_paths:list):
    """
    Creates the manifest file given the metadata file.
    """
    # Load metadata file; this is the one aggregating info from all datasets
    df_metadata = pd.read_csv(metadata_path, sep=';')
    # Split the metadata file into train,valid,and test files
    df_train = df_metadata[df_metadata['split']==0]
    dataframe_to_json(df_train,manifest_paths[0])
    df_valid = df_metadata[df_metadata['split']==1]
    dataframe_to_json(df_valid,manifest_paths[1])
    df_test = df_metadata[df_metadata['split']==2]
    dataframe_to_json(df_test,manifest_paths[2])

    logger.info(f"{manifest_paths} successfully created!")


def dataframe_to_json(df,save_path):
    # we now build JSON examples 
    examples = {}
    for _, row in df.iterrows():
        utt_id = row['Uid'] # returns the name (without extension) E.g., '00000','00010'
        examples[utt_id] = {"ID": utt_id,
                            "file_path": row['voice-path-new'],
                            "emd_path": row['Embedding_path'],
                            "label": row['label'],
                            }
        
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, "w") as f:
        json.dump(examples, f, indent=4)

    return examples


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def unzip_audio_file(destination, audio_archive_path):
    """
    Unzip the compressed audio folder.
    """
    if not os.path.exists(audio_archive_path):
        raise ValueError("Audio zip file not found. Please refer to prep.ipynb first to prepare the zip file.")
    shutil.unpack_archive(audio_archive_path, os.path.dirname(destination)) # this will create a folder called 'TASK1-VOICE' inside of the 'wav' folder
    

class COVIDforGAN(Dataset):

    def __init__(self, manifest_file):
        """_summary_

        Args:
            manifest_file (str): file name of the manifest file (XXX.json)
        """

        with open(manifest_file) as f:
            self.manifest_dict = json.load(f)

    def __len__(self):
        return len(self.manifest_dict)

    def __getitem__(self, idx):
        idx_new = list(self.manifest_dict.keys())[idx]
        embedding = self.load_emd(self.manifest_dict[idx_new]['emd_path'])
        label = torch.tensor(int(self.manifest_dict[idx_new]['label'])).view(1,)
        return embedding, label

    def load_emd(self, emd_file:str):
        """_summary_
        Load the pre-extracted speaker embedding file. 
        As this will save us some time than loading them again during iterations.

        Args:
            emd_file (str): path to the embedding file
        """
        x = torch.load(emd_file)
        return x