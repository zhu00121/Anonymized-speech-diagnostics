"""
Pipeline description:
This system is composed of an ECAPA-TDNN model. 
It is a combination of convolutional and residual blocks. 
The embeddings are extracted using attentive statistical pooling. 
The system is trained with Additive Margin Softmax Loss. 
Speaker Verification is performed using cosine distance between speaker embeddings.

It is trained on Voxceleb 1+ Voxceleb2 training data.
"""

import os,sys
import numpy as np
from Utils import util
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition


def get_embeddings(audio_path:str):

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    signal, _ = torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)

    return embeddings


def verify_speaker(audio_og_path:str):

    head, tail = os.path.split(audio_og_path)
    sample_id = util.remove_suffix(tail,'.wav')
    sample_id = util.remove_suffix(sample_id,'.flac')
    audio_mcadams_path = head + sample_id + '_mcadams.wav'
    audio_ss_path = head + sample_id + '_ss.wav'

    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./Local/ASV/pretrained_models/spkrec-ecapa-voxceleb")
    
    choices = [audio_og_path,audio_mcadams_path,audio_ss_path]
    score_all = []
    prediction_all = []

    for i in choices:
        for j in choices:
            score, prediction = verification.verify_files(i,j)
            score_all.append(score)
            prediction_all.append(prediction)
    
    score_all = np.asarray(score_all).squeeze()
    prediction_all = np.asarray(score_all).squeeze()

    return score_all, prediction_all

# TODO: for each dataset, loop trough speakers and calculate EER

    # if print_path == None:
    #     print_path = os.path.join(os.path.split(save_path)[0],'summary.txt')

    # print('-----')
    # util.print_to_file(print_path,'-----')
    # if notes is not None:
    #     util.print_to_file(print_path,notes)
    # print('True AUC-ROC: %.3f'%(result['ROC_TRUE']))
    # util.print_to_file(print_path,'True AUC-ROC: %.3f'%(result['ROC_TRUE']))
    # print('True UAR: %.3f'%(result['UAR_TRUE']))
    # util.print_to_file(print_path,'True UAR: %.3f'%(result['UAR_TRUE']))
    # print('Average AUC-ROC: %.3f'%(result['ROC_AVE'][0]))
    # util.print_to_file(print_path,'Average AUC-ROC: %.3f'%(result['ROC_AVE'][0]))
    # print('CI on AUC-ROC: %.3f-%.3f'%(result['ROC_AVE'][1],result['ROC_AVE'][2]))
    # util.print_to_file(print_path,'CI on AUC-ROC: %.3f-%.3f'%(result['ROC_AVE'][1],result['ROC_AVE'][2]))
    # print('Average UAR: %.3f'%(result['UAR_AVE'][0]))
    # util.print_to_file(print_path,'Average UAR: %.3f'%(result['UAR_AVE'][0]))
    # print('CI on UAR: %.3f-%.3f'%(result['UAR_AVE'][1],result['UAR_AVE'][2]))
    # util.print_to_file(print_path,'CI on UAR: %.3f-%.3f'%(result['UAR_AVE'][1],result['UAR_AVE'][2]))


