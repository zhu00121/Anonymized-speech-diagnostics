"""
This script is used to generate experimental set-up files (.json).
"""

import sys,os
sys.path.append("./Local/")
from Utils import util


class setup():

    def __init__(self,kwargs):
        self.condition = {'og':['og-og'],
                         'ignorant':['og-ss','og-mcadams','og-pros'],
                         'semi-informed':['mcadams-ss','ss-mcadams','mcadams-pros','ss-pros','pros-mcadams','pros-ss'],
                         'informed':['mcadams-mcadams','ss-ss','pros-pros'],
                         'augmented':[
                            'og_og-ss','og_ss-ss','og_mcadams-ss','og_pros-ss',
                            'og_og-mcadams','og_mcadams-mcadams','og_ss-mcadams','og_pros-mcadams',
                            'og_og-pros', 'og_mcadams-pros', 'og_ss-pros', 'og_pros-pros',
                            'ss_ss-mcadams','ss_mcadams-mcadams','ss_pros-mcadams',
                            'mcadams_ss-ss','mcadams_mcadams-ss','mcadams_pros-ss',
                            'mcadams_ss-pros']
                         }
        self.datasets = ['CSS','DiCOVA2','Cambridge']
        self.ano_method = ['og','mcadams','ss','pros']
        self.pipeline = ['openSMILE_pca-svm','openSMILE_svm','msr_pca-svm','msr_svm','logmelspec_bilstm']

        self.exp_root_path = kwargs.get('exp_config_root','./Config/exp_config')
        if not os.path.exists(self.exp_root_path):
            os.mkdir(self.exp_root_path)
        
        self.model_config_dir = './Config/model_config'
        self.metadata_path = kwargs['metadata_path']

    
    def _create_exp_subcondition(self,subcondition,save_dir,**options):
        """
        Create experimental setup file for a system
        """
        print('Subcondition %s'%(subcondition))
        # sanity check
        _source_ano, _target_ano = subcondition.split('-')
        # assert _source_ano in self.ano_method and _target_ano in self.ano_method, "unknown subcondition: %s"%(subcondition)
        
        # define parameters
        for system in self.pipeline:
            _system_path = os.path.join(save_dir,system)
            if not os.path.exists(_system_path):
                os.mkdir(_system_path)
            _feat,_clf = system.split('_')
            _clf_config = util.load_json(os.path.join(self.model_config_dir,'%s_config')%(_clf))

            _source = options.get('source')
            _target = options.get('target')

            # if given a specified training and test dataset, then:
            if _source is not None and _target is not None:
                source_anos = _source_ano.split('_')
                sources = _source.split('_')
                exp_param = {
                            'feat':_feat,
                            'train_set':sources,
                            'train_ano_mode':source_anos,
                            'test_set':_target,
                            'test_ano_mode':_target_ano,
                            'train_metadata_path':[self.metadata_path['%s'%(sources[i])]['%s'%(source_anos[i])] for i in range(len(source_anos))],
                            'test_metadata_path':self.metadata_path['%s'%(_target)]['%s'%(_target_ano)],
                            'pipeline_kwargs':_clf_config['pipeline_kwargs']
                            }
        
                # save experimental set-up as .json file
                filename = '%s_%s_%s_%s_%s'%(_source,_source_ano,_target,_target_ano,system)
                util.save_as_json(exp_param,os.path.join(_system_path,filename))
            
            elif _source is None and _target is None:
                # try every combination
                for _source in self.datasets:
                    if _source == 'DiCOVA2':
                        _clf_config['pipeline_kwargs']['pds'] = 'no'
                    elif _source != 'DiCOVA2':
                        _clf_config['pipeline_kwargs']['pds'] = 'yes'
                    for _target in self.datasets:
                        _train_metadata_path = self.metadata_path['%s'%(_source)]['%s'%(_source_ano)]

                        exp_param = {
                                    'feat':_feat,
                                    'train_set':_source,
                                    'train_ano_mode':_source_ano,
                                    'test_set':_target,
                                    'test_ano_mode':_target_ano,
                                    'train_metadata_path':self.metadata_path['%s'%(_source)]['%s'%(_source_ano)],
                                    'test_metadata_path':self.metadata_path['%s'%(_target)]['%s'%(_target_ano)],
                                    'pipeline_kwargs':_clf_config['pipeline_kwargs']
                                    }
            
                        # save experimental set-up as .json file
                        filename = '%s_%s_%s_%s_%s'%(_source,_source_ano,_target,_target_ano,system)
                        util.save_as_json(exp_param,os.path.join(_system_path,filename))

        print('Finished set-up for current subcondition %s'%(subcondition))


    def _create_exp_condition(self,condition:str):
        """
        Set up for one condition.
        """
        print('---Set up for condition %s---'%(condition))
        _condition_path = os.path.join(self.exp_root_path,condition)
        if not os.path.exists(_condition_path):
            os.mkdir(_condition_path)

        for i in self.condition[condition]:
            _subcondition_path = os.path.join(_condition_path,i)
            if not os.path.exists(_subcondition_path):
                os.mkdir(_subcondition_path)

            if condition  == 'augmented':
                # self._create_exp_subcondition(i,_subcondition_path,**{'source':'CSS_Cambridge','target':'DiCOVA2'})
                self._create_exp_subcondition(i,_subcondition_path,**{'source':'DiCOVA2_CSS','target':'DiCOVA2'})
                self._create_exp_subcondition(i,_subcondition_path,**{'source':'DiCOVA2_Cambridge','target':'DiCOVA2'})
                # self._create_exp_subcondition(i,_subcondition_path,**{'source':'Cambridge_CSS','target':'DiCOVA2'})
                # self._create_exp_subcondition(i,_subcondition_path,**{'source':'Cambridge_CSS','target':'CSS'})
                # self._create_exp_subcondition(i,_subcondition_path,**{'source':'Cambridge_CSS','target':'Cambridge'})
            elif condition != 'augmented':
                self._create_exp_subcondition(i,_subcondition_path)

        print('Finished set-up for condition %s'%(condition))
            
    
    def run(self,mode:str='Default'):
        """
        Set up all conditions.
        """
        assert mode in ['og','ignorant','semi-informed','informed','augmented']
        print('--------Creating experimental set-up files--------')
        if mode != 'Default':
            self._create_exp_condition(mode)
        elif mode == 'Default':
            for c in self.condition.keys():
                self._create_exp_condition(c)
        print('--------All experimental set-up files have been created--------')


# %% 
if __name__ == '__main__':

    kwargs = {'metadata_path':{
                             'CSS':{
                                   'og':'/mnt/d/projects/COVID-datasets/CSS/label/metadata_og.csv',
                                   'mcadams':'/mnt/d/projects/COVID-datasets/CSS/label/metadata_mcadams.csv',
                                   'ss':'/mnt/d/projects/COVID-datasets/CSS/label/metadata_ss.csv',
                                   'pros':'/mnt/d/projects/COVID-datasets/CSS/label/metadata_pros.csv'
                                   },
                             'DiCOVA2':{
                                   'og':'/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_og.csv',
                                   'mcadams':'/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_mcadams.csv',
                                   'ss':'/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_ss.csv',
                                   'pros': '/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_pros.csv'
                                   },
                             'Cambridge':{
                                   'og':'/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_og.csv',
                                   'mcadams':'/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_mcadams.csv',
                                   'ss':'/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_ss.csv',
                                   'pros':'/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_pros.csv'
                                   },
                             }}

    setup(kwargs).run(mode='ignorant')
    setup(kwargs).run(mode='informed')
    setup(kwargs).run(mode='augmented')