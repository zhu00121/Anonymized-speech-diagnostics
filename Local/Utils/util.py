import json
import os
import pickle as pkl
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def print_to_file(file_path:str,print_content:str):
    """
    Save print content to a .txt file for quick accessing experiment results.
    """
    with open(file_path, "a") as f:
        print(print_content, file=f)

    return 0


def save_as_json(data,save_path):
    """
    Save results as .json file.
    """
    # assert not os.path.exists(save_path), "File existing with the same name."
    with open(save_path, "w") as outfile:
        json.dump(data, outfile)
    
    return 0


def load_json(save_path:str):
    """
    Load a saved .json file.
    """
    if not os.path.exists(save_path):
        print(save_path)

    assert os.path.exists(save_path), "json file does not exist!"
    with open(save_path, 'r') as f:
        saved_json = json.load(f)
    return saved_json


def save_as_pkl(save_path:str,variable):

    """
    Save result as a pickle file.
    """
    # assert not os.path.exists(save_path), 'Existing file with same name.'
    
    with open(save_path, "wb") as outfile:
        pkl.dump(variable, outfile, pkl.HIGHEST_PROTOCOL)
        
    return 0


def load_pkl(save_path:str):

    """
    Load saved pickle file.
    """
    assert os.path.exists(save_path), "pkl file does not exist in %s!"%(save_path)
    with open(save_path, "rb") as infile:
        saved_fea = pkl.load(infile)

    return saved_fea


def load_model(clf_path:str,mode:str):

    if mode == 'sklearn':
        clf = load_pkl(clf_path)
    elif mode == 'pytorch':
        clf = torch.load(clf_path)
    return clf


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def convertType(val):
	def subutil(val):
		try: 
			val = int(val)
		except:
			try:
				val = float(val)
			except:
				if val in ['True', 'TRUE', 'true']:
					val = True
				elif val in ['False','FALSE','false']:
					val = False
				elif val in ['None']:
					val = None
				else:
					val = val
		return val

	if ',' in val:
		val = val.split(',')
		val = [subutil(item) for item in val]
	else:
		val = subutil(val)
	return val


def load_feat(metadata_path:str,feat_name:str,split:str):
    """
    Load features and labels using file paths stored in the metadata.csv file.
    """
    assert feat_name in ['openSMILE','logmelspec','msr','mtr', 'mtr_v2','mtr_v3'], "Feature type is not supported"

    print("Loading data...")
    feat_col = 'feature_path_'+feat_name
    df_md = pd.read_csv(metadata_path)
    # check if the input split names exist (e.g., valid/validation)
    if (not (df_md['split'].eq(split)).any()) and (split == 'valid'):
        split = 'validation'
    df_md = df_md[df_md['split']==split]
    feat_path_list = df_md[feat_col].to_list()
    label_list = df_md['label'].to_numpy()
    label_list = label_list.reshape(label_list.shape[0],1)
    assert len(feat_path_list) == len(label_list), "number of features is not equal to number of labels"
    num_sample = len(feat_path_list)
    feat_all = []
    for idx in tqdm(range(num_sample)):
        feat = load_pkl(feat_path_list[idx])
        feat_all.append(feat)

    feat_all = np.asarray(feat_all).squeeze()

    # sanity check on feature shape
    if feat_name == 'logmelspec':
        assert feat_all.ndim == 3, "msr and logmelspec features should be 3-dimensional"
    elif feat_name == 'msr' or feat_name == 'openSMILE ':
        assert feat_all.ndim == 2, "openSMILE and msr features should be 2-dimensional"
    elif feat_name == 'mtr':
        assert feat_all.ndim == 4, "mtr features should be 4-dimensional"

    return feat_all, label_list
