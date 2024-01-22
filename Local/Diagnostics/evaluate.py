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
    source_ano, target_ano = subcondition.split('-')
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
    datasets = ['Cambridge','CSS','DiCOVA2','DiCOVA2_CSS','DiCOVA2_Cambridge']
    print('---')
    print('Pipeline: %s-%s'%(feat,classifier))
    for source in datasets:
        for target in datasets:
            print('Source dataset:%s | Anonymization:%s'%(source,source_ano))
            print('Target dataset:%s | Anonymization:%s'%(target,target_ano))
            exp_name = '%s_%s_%s_%s_%s_%s'%(source,source_ano,target,target_ano,feat,classifier)
            try:
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
                if not os.path.exists(save_to_dir):
                    os.makedirs(save_to_dir)

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

            except: 
                print(exp_name + ' not found; pass to next task.')
                pass

    print('All experiments for pipeline: %s-%s are finished.'%(feat,classifier))
    
    return 0


class eva_main():

    def __init__(self):

        self.condition = {
        'ignorant':['og-mcadams','og-ss','og-pros'],
        'semi-informed':['mcadams-pros','mcadams-ss','pros-mcadams','pros-ss','ss-mcadams','ss-pros'],
        'informed':['mcadams-mcadams','ss-ss','pros-pros'],
        'augmented':['og_mcadams-ss', 'og_ss-mcadams',
                     'og_pros-mcadams', 'og_pros-ss',
                     'ss_pros-mcadams','ss_mcadams-mcadams',
                     'mcadams_ss-ss','mcadams_pros-ss',
                     ],
        }

        self.pipeline = ['openSMILE_svm',
                         'msr_svm', 
                         'logmelspec_bilstm']

    def _eva_per_subcondition(self, condition, subcondition):
        
        print('----- Evaluate all systems under subcondition %s ------'%(subcondition))

        for system in self.pipeline:
            _feat, _classifier = system.split('_')
            eva_per_system(
                subcondition=subcondition,\
                feat=_feat,
                classifier=_classifier,
                subcondition_config_dir='./Config/exp_config/%s/%s'%(condition,subcondition),
                save_to_dir='./Results/performance/%s/%s/%s'%(condition,subcondition,system)
            )

        print('----- Experiments for sub-condition %s are finished -----'%(subcondition))
        
        return 0

    def _eva_per_condition(self, condition):
        
        print('----- Evaluate all sub-conditions under the condition %s ------'%(condition))

        for sub in self.condition[condition]:
            self._eva_per_subcondition(condition,sub)
        
        print('----- Experiments for condition %s are finished -----'%(condition))

        return 0

    def main(self):

        print('----- Start evaluation -----')

        for con in self.condition.keys():
            self._eva_per_condition(con)

        print('----- Finish evaluation -----')
        
        return 0


# %%
if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    eva_main().main()