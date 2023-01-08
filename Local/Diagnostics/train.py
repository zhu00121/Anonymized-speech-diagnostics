"""
Train a diagnostics system/model.
This script can be used to train four systems included in this study.
"""

import os,sys
sys.path.append("./Local/")
import numpy as np
import pandas as pd
from Utils import util
import ML_func
import crnn
import bilstm
import torch
import torch.optim as optim
import random
import warnings

def train_uni(feat:str,train_set:str,ano_mode:str,metadata_path:str,clf_dir:str=None,**clf_kwargs):
    """
    Given training (and validation) data, train a diagnostics system/model. Trained model is then saved
    for further testing.
    """
    # sanity check
    assert feat in ['openSMILE','logmelspec','msr','mtr'], "Input mode is not supported"
    assert train_set in ['CSS','DiCOVA2','Cambridge'], "Input train set is not found"
    assert ano_mode in ['og','mcadams'] # TODO: add more anonymization modes
    if not ano_mode in metadata_path: warnings.warn("Inconsistency between anonymization mode and metadata file name. Ensure the correct metadata file is used.")

    # model saving path
    if clf_dir is None:
        clf_dir = './Results/Pretrained'
    folder_name = os.path.join(clf_dir,'%s_%s_%s_%s'%(train_set,ano_mode,feat,clf_kwargs['model']))
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    # main loops for training
    # for 1-D features
    if feat == 'openSMILE' or feat == 'msr':
    
        # load data
        print('Load data...')
        train_data,train_label = util.load_feat(metadata_path=metadata_path,feat_name=feat,split='train')

        # use pre-defined valid set
        if clf_kwargs['pds'] == 'yes':
            valid_data,valid_label = util.load_feat(metadata_path=metadata_path,feat_name=feat,split='valid')
            print('A pre-defined train-validation split will be used.')
        elif clf_kwargs['pds'] == 'no':
            print('Only training set is defined. N-fold CV will be performed.')

        # train shallow ML models
        if clf_kwargs['pds'] == 'yes':
            train_data = np.concatenate((train_data,valid_data),axis=0)
            train_label = np.concatenate((train_label,valid_label),axis=0)
            split_idx = [0 if i >= (len(train_label)-len(valid_label)) else -1 for i in range(len(train_label))]
            clf_kwargs['split_index'] = clf_kwargs.get('split_idx',split_idx)
            clf_kwargs['model'] = clf_kwargs.get('model','svm')

        elif clf_kwargs['pds'] == 'no':
            clf_kwargs['n_splits'] = clf_kwargs.get('n_splits',3)
            clf_kwargs['n_repeats'] = clf_kwargs.get('n_repeats',3)
            clf_kwargs['model'] = clf_kwargs.get('model','svm')

        # start training
        clf = ML_func.train_model(train_data,train_label,clf_kwargs)

        # save trained models
        util.save_as_pkl(os.path.join(folder_name,'clf.pkl'), clf)
        print('Trained model is saved as %s'%(os.path.join(folder_name,'clf.pkl')))

    elif feat == 'logmelspec' or feat == 'mtr':

        if feat == 'logmelspec':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # set RNG for reproducibility
            ML_func.set_seed(clf_kwargs['RNG'],device)
            # define classifier, optimizer and loss function
            model = bilstm.BiLSTM(clf_kwargs).to(device)
            optimizer=optim.Adam(model.parameters(), lr=clf_kwargs['lr'], weight_decay=clf_kwargs['weight_decay'], amsgrad=True)
            criterion = torch.nn.BCEWithLogitsLoss()

        elif feat == 'mtr':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # set RNG for reproducibility
            ML_func.set_seed(clf_kwargs['RNG'],device)
            # define classifier, optimizer and loss function
            model = crnn.crnn_cov_3d(clf_kwargs).to(device)
            optimizer = optim.Adam(list(model.parameters()),lr=clf_kwargs['lr'],weight_decay=clf_kwargs['weight_decay'])
            criterion = torch.nn.BCEWithLogitsLoss()

        # start training
        test_score_list, std_list = ML_func.train_main(model,device,optimizer,criterion,
                           train_set,feat,clf_kwargs['batch_sizes'],
                           folder_name,metadata_path,
                           mode='joint',filters=clf_kwargs.get('filters',None))

    # save model kwargs
    util.save_as_json(clf_kwargs,os.path.join(folder_name,'clf_kwargs'))
    print('Model parameters are saved in %s'%(os.path.join(folder_name,'clf_kwargs')))

    return test_score_list, std_list


# %%
if __name__ == '__main__':

    # pipeline_kwargs = {'model':'pca-svm','pds':'yes','n_components':100}

    # kwargs = {
    #         'feat':'openSMILE', \
    #         'train_set': 'CSS', \
    #         'train_ano_mode':'og', \
    #         'test_set': 'CSS', \
    #         'test_ano_mode': 'og', \
    #         'metadata_path': '/mnt/d/projects/COVID-datasets/CSS/label/metadata_og.csv', \
    #         'pipeline_kwargs': pipeline_kwargs}
    
    # util.save_as_json(kwargs,'./Config/exp_config/og/CSS_og_CSS_og_openSMILE_pca-svm')
    torch.cuda.empty_cache()
    clf_kwargs = util.load_json('./Config/model_config/bilstm_config')
    test_score, _std = train_uni(feat='logmelspec',\
                        train_set='CSS',
                        ano_mode='og',
                        metadata_path='/mnt/d/projects/COVID-datasets/CSS/label/metadata_og.csv',
                        **clf_kwargs['pipeline_kwargs'])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(test_score[2:])
    plt.plot(_std[1:])
    plt.plot(np.diff(_std,n=1))
    plt.plot(_std[1:]+np.abs(np.diff(_std,n=1)))
    plt.show()