"""
Testing a trained model in terms of diagnostics performance.
"""

import os,sys
sys.path.append("./Local/")
import numpy as np
import pandas as pd
from Utils import util
import ML_func


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
        clf = util.load_model(os.path.join(folder_name,'clf.h5'),mode='pytorch')
    pt_clf_kwargs = util.load_json(os.path.join(folder_name,'clf_kwargs'))

    # load test data
    print('Load test data (%s-%s)'%(test_set,test_ano_mode))
    test_data,test_label = util.load_feat(metadata_path=metadata_path,feat_name=feat,split='test')

    # define where the results are saved 
    if res_dir is None:
        res_dir = './Results/performance'
    result_path = os.path.join(res_dir,'%s_%s_%s_%s_%s_%s.pkl')%(train_set,train_ano_mode,test_set,test_ano_mode,feat,model)
    notes = '%s-%s-%s-%s-%s-%s'%(train_set,train_ano_mode,test_set,test_ano_mode,feat,model)

    print('Evaluating diagnostics performance with 1000 bootstrapping...')
    # for 1-D features
    if feat == 'openSMILE' or feat == 'msr':
        result = ML_func.eva_model(test_data,test_label,clf,save_path=result_path,notes=notes) # results are automatically saved
    
    # TODO: other features

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