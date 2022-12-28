"""
Train a diagnostics system/model.
This script can be used to train four systems included in this study.
"""

def train_model(mode:str,):

    # sanity check
    assert mode in ['openSMILE-SVM','logmelspec-BiLSTM','msr-SVM','mtr-CRNN'], "Input mode is not supported"

    # main loops for training
    # for those two using SVM as classifiers
    if 'SVM' in mode:
        # TODO

    elif mode == 'logmelspec-BiLSTM':
        # TODO
    
    elif mode == 'mtr-CRNN':
        # TODO
    
    # save trained models

    return 0
