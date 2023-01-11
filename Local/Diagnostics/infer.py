"""
Testing a trained model in terms of diagnostics performance.
"""

import os,sys
sys.path.append("./Local/")
import numpy as np
import pandas as pd
import torch
from Utils import util
import ML_func
import bilstm
import crnn
import covid_dataset
from torch.utils.data import DataLoader


def infer_uni(feat:str,train_set:str,train_ano_mode:str,test_set:str,test_ano_mode:str,model:str,metadata_path:str,clf_dir:str=None,res_dir:str=None):

    # sanity check
    assert feat in ['openSMILE','logmelspec','msr','mtr'], "Input mode is not supported"
    assert train_set in ['CSS','DiCOVA2','Cambridge'], "Input train set is not found"
    assert test_set in ['CSS','DiCOVA2','Cambridge'], "Input test set is not found"
    assert train_ano_mode in ['og','mcadams'], "Training anomyzation mode is not found" # TODO: add more anonymization modes
    assert test_ano_mode in ['og','mcadams'], "Test anomyzation mode is not found" # TODO: add more anonymization modes
    
    # load pre-trained classifier
    print('Load pre-trained classifier (%s-%s)'%(train_set,train_ano_mode))
    if clf_dir is None:
        clf_dir = './Results/Pretrained'
    folder_name = os.path.join(clf_dir,'%s_%s_%s_%s'%(train_set,train_ano_mode,feat,model))
    if model == 'svm' or model == 'pca-svm':
        clf = util.load_model(os.path.join(folder_name,'clf.pkl'),mode='sklearn')
    elif model == 'bilstm' or model == 'crnn':
        clf = util.load_model(os.path.join(folder_name,'clf.pt'),mode='pytorch')
    pt_clf_kwargs = util.load_json(os.path.join(folder_name,'clf_kwargs'))

    # define where the results are saved 
    if res_dir is None:
        res_dir = './Results/performance'
    result_path = os.path.join(res_dir,'%s_%s_%s_%s_%s_%s.pkl')%(train_set,train_ano_mode,test_set,test_ano_mode,feat,model)
    notes = '%s-%s-%s-%s-%s-%s'%(train_set,train_ano_mode,test_set,test_ano_mode,feat,model)

    print('Evaluating diagnostics performance with 1000 bootstrapping...')
    # for 1-D features
    if feat == 'openSMILE' or feat == 'msr':
        # load test data
        print('Load test data (%s-%s)'%(test_set,test_ano_mode))
        test_data,test_label = util.load_feat(metadata_path=metadata_path,feat_name=feat,split='test')
        # evaluate
        result = ML_func.eva_model(test_data,test_label,clf,save_path=result_path,notes=notes) # results are automatically saved
    
    # TODO: test the following code
    if feat == 'logmelspec' or 'mtr' in feat:
        # load model architecture and pre-trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model == 'bilstm': pt_clf = bilstm.BiLSTM(pt_clf_kwargs).to(device)
        elif model == 'crnn': pt_clf = crnn.crnn_cov_3d(pt_clf_kwargs.to(device))
        pt_clf.load_state_dict(clf)
        # load test data (/dataloader)
        test_data = covid_dataset._covid_dataset(dataset=test_set,split='test',feat=feat,metadata_path=metadata_path)
        test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False)
        # evaluate
        result = ML_func.eva_model(pt_clf=pt_clf,framework='pytorch',test_dataloader=test_loader,device=device,save_path=result_path,notes=notes)

    return 0


# %%
# if __name__ == '__main__':

    # exp_kwargs_dir = './Config/exp_config/CSS_og_CSS_og_openSMILE_pca-svm'
    # kwargs = util.load_json(exp_kwargs_dir)
    
    # infer_uni(
    #          kwargs['feat'],kwargs['train_set'],kwargs['train_ano_mode'],\
    #          kwargs['test_set'],kwargs['test_ano_mode'],kwargs['pipeline_kwargs']['model'],\
    #          kwargs['metadata_path']
    #          )
    
# util.save_as_json(kwargs,exp_kwargs)