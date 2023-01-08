# TODO: Train and test a diagnostics system and save the performance.

import os,sys
sys.path.append("./Local/")
import numpy as np
import pandas as pd
from Utils import util
from Diagnostics import train
from Diagnostics import infer


def eva_per_system(
                  subcondition:str,
                  feat:str,classifier:str,
                  default_clf_path:str=None,
                  subcondition_config_dir:str=None,
                  save_to_dir:str=None
                  ):
    """
    Given a subcondition, evaluate a diagnostic system with within-dataset test and cross-dataset test.
    """
    print('Setting up experiments...')
    # sanity check
    # assert condition in ['og','ignorant','semi-informed','informed','augmented']
    source_ano, target_ano = subcondition.split('-')
    assert source_ano in ['og','mcadams','ss','mcadams_ss'] # TODO: add more anonymization modes
    assert target_ano in ['og','mcadams','ss','mcadams_ss'] # TODO: add more anonymization modes
    assert feat in ['openSMILE','logmelspec','msr','mtr'], "Input feature type is not supported"
    assert classifier in ['svm','pca-svm','bilstm','crnn']

    # specify root path
    assert os.path.exists(subcondition_config_dir), "Check configuration directory for this subcondition"

    # trace back to the corresponding folder which contains the experimental set-up details
    dir_0 = os.path.join(subcondition_config_dir,'%s_%s'%(feat,classifier))
    if not os.path.exists(dir_0):
        print('Experimental set-up folder for system %s_%s does not exist. Check input DIR.'%(feat,classifier))

    # set-up experiments for within- and cross-dataset evaluation
    # TODO: add sub-condition
    datasets = ['Cambridge','CSS','DiCOVA2']
    print('---')
    print('Pipeline: %s-%s'%(feat,classifier))
    for source in datasets:
        for target in datasets:
            print('Source dataset:%s | Anonymization:%s'%(source,source_ano))
            print('Target dataset:%s | Anonymization:%s'%(target,target_ano))
            exp_name = '%s_%s_%s_%s_%s_%s'%(source,source_ano,target,target_ano,feat,classifier)
            exp_kwargs = util.load_json(os.path.join(dir_0,exp_name))

            # check if a pre-trained model exists, if not, train one
            if default_clf_path is None:
                default_clf_path = './Results/Pretrained/'
            print('---Training phase---')
            if os.path.exists(os.path.join(default_clf_path,'%s_%s_%s_%s'%(source,source_ano,feat,classifier))):
                print('Found pre-trained model')
            elif not os.path.exists(os.path.join(default_clf_path,'%s_%s_%s_%s'%(source,source_ano,feat,classifier))):
                print('Pre-trained model not found in %s. Start training one from scratch...'%(default_clf_path))
                clf = train.train_uni(
                                    exp_kwargs['feat'],\
                                    exp_kwargs['train_set'],\
                                    exp_kwargs['train_ano_mode'],\
                                    exp_kwargs['train_metadata_path'],\
                                    clf_dir = default_clf_path,\
                                    **exp_kwargs['pipeline_kwargs']
                                    )
            print('---Inference phase---')
            # sanity check on results saving path
            if os.path.exists(save_to_dir):
                save_to_dir = '%s/%s_%s'%(subcondition,feat,classifier)
                if not os.path.exists(save_to_dir):
                    os.mkdir(save_to_dir)

            infer.infer_uni(
                            exp_kwargs['feat'],
                            exp_kwargs['train_set'],
                            exp_kwargs['train_ano_mode'],
                            exp_kwargs['test_set'],
                            exp_kwargs['test_ano_mode'],
                            classifier,
                            exp_kwargs['test_metadata_path'],
                            default_clf_path,
                            res_dir=save_to_dir
                            )
            print('---')
    print('All experiments for pipeline: %s-%s are finished.'%(feat,classifier))
    
    return 0


def eva_per_subcondition(condition:str):




# TODO: complete the following function
def eva_per_condition(condition:str,source_ano:str,target_ano:str,feat:str,classifier:str,pt:bool,parent_dir:str=None):
    """
    Given an anonymization mode (i.e., original/ignorant/semi-informed/informed/augmented), evaluate
    all diagnostics systems.
    """

    return 0


# TODO: complete the following function
def eva_all_condition(parent_dir:str):
    """
    Evaluate all conditions for all diagnostics systems.
    """

    return 0


# %%
if __name__ == '__main__':

    # test function with og condition
    eva_per_system(
        subcondition='og-og',
        feat='msr',
        classifier='pca-svm',
        subcondition_config_dir='./Config/exp_config/og/og-og',
        save_to_dir='./Results/performance/og/og-og'
    )