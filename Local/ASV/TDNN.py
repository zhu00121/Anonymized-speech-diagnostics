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
sys.path.append('./Local/')
import numpy as np
import pandas as pd
from tqdm import tqdm
from Utils import util
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition


def get_embeddings(audio_path:str):

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    signal, _ = torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)

    return embeddings


def verify_speaker(audio_og_path:str,verification):

    head, tail = os.path.split(audio_og_path)
    sample_id = util.remove_suffix(tail,'.wav')
    sample_id = util.remove_suffix(sample_id,'.flac')
    audio_mcadams_path = os.path.join(head, sample_id) + '_mcadams.wav'
    audio_ss_path = os.path.join(head, sample_id) + '_ss.wav'
    
    choices = [audio_og_path,audio_mcadams_path,audio_ss_path]
    score_all = []
    prediction_all = []

    for i in choices:
        for j in choices:
            score, prediction = verification.verify_files(i,j)
            score_all.append(score.detach().cpu().to_numpy())
            prediction_all.append(prediction.detach().cpu().to_numpy())
    
    score_all = np.asarray(score_all).squeeze()
    prediction_all = np.asarray(score_all).squeeze()

    return score_all, prediction_all


class verify_main():

    def __init__(self):
        
        self.metadata_path = [
            '/mnt/d/projects/COVID-datasets/CSS/label/metadata_v2.csv',\
            '/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_v2.csv',
            '/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_v2.csv'
            ]
        self.verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./Local/ASV/pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"} )
        
    def start_verify(self):
        
        res_score = []
        res_preds = []
        for dataset in self.metadata_path:
            df = pd.read_csv(dataset) # load metadata
            all_audio_path = df['audio_path'].tolist()
            scores = []
            preds = []
            for i in tqdm(all_audio_path):
                try:
                    s,p = verify_speaker(i,self.verifier)
                    scores.append(s)
                    preds.append(p)
                except:
                    pass
            
            ave_score = np.mean(np.asarray(scores),axis=0)
            ave_preds = np.mean(np.asarray(preds),axis=0)
            res_score.append(ave_score)
            res_preds.append(ave_preds)
        
        return res_score, res_preds


# %%
if __name__ == '__main__':

    ASV_result = verify_main().start_verify()