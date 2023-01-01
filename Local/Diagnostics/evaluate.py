# TODO: Train and test a diagnostics system and save the performance.

import os,sys
sys.path.append("./Local/")
import numpy as np
import pandas as pd
from Utils import util
import train
import infer


def eva_per_system(condition:str,source_ano:str,target_ano:str,feat:str,classifier:str,pt:bool,parent_dir:str=None):
    """
    Given an anonymization mode (i.e., original/ignorant/semi-informed/informed/augmented), evaluate
    a diagnostic system with within-dataset test and cross-dataset test.
    """
    print('Setting up experiments...')
    # sanity check
    assert condition in ['og','ignorant','semi-informed','informed','augmented']
    assert source_ano in ['og','mcadams'] # TODO: add more anonymization modes
    assert target_ano in ['og','mcadams'] # TODO: add more anonymization modes
    assert feat in ['openSMILE','logmelspec','msr','mtr'], "Input feature type is not supported"
    assert classifier in ['svm','pca-svm','bilstm','crnn']

    # specify root path
    if parent_dir == None or not os.path.exists(parent_dir):
        parent_dir = './Config/exp_config'

    # trace back to the corresponding folder which contains the experimental set-up details
    dir_0 = os.path.join(parent_dir,'%s')%(condition)

    # set-up experiments for within- and cross-dataset evaluation
    datasets = ['CSS','DiCOVA2','Cambridge']
    print('----------Start experiments for condition %s----------'%(condition))
    print('Pipeline: %s-%s'%(feat,classifier))
    for source in datasets:
        for target in datasets:
            print('Source dataset:%s | Anonymization:%s'%(source,source_ano))
            print('Target dataset:%s | Anonymization:%s'%(target,target_ano))
            exp_name = '%s_%s_%s_%s_%s_%s'%(source,source_ano,target,target_ano,feat,classifier)
            exp_kwargs = util.load_json(os.path.join(dir_0,exp_name))

            # TODO: train system/model or load a pre-trained system/model
            # TODO: check if a pretrained model exists, if not, training one
            print('---Training phase---')
            if pt:
                # load pre-trained model
                clf = util.load_json()
            elif not pt:
                clf = train.train_uni(exp_kwargs)

            # TODO: test system/model (results saved automatically)
            print('---Inference phase---')
            infer.infer_uni(exp_kwargs)
            print('---------------------')
    
    # TODO: Save results to sub-folders in Results/performance/.

    return 0


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