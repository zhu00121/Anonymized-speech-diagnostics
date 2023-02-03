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
from scipy import spatial


def get_embeddings(audio_path:str,classifier):

    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    signal, _ = torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)

    return embeddings


def verify_speaker(audio_og_path:str,classifier):

    head, tail = os.path.split(audio_og_path)
    sample_id = util.remove_suffix(tail,'.wav')
    sample_id = util.remove_suffix(sample_id,'.flac')
    audio_mcadams_path = os.path.join(head, sample_id) + '_mcadams.wav'
    audio_ss_path = os.path.join(head, sample_id) + '_ss.wav'
    
    choices = [[audio_og_path,audio_mcadams_path],[audio_og_path,audio_ss_path],[audio_mcadams_path,audio_ss_path],[audio_og_path,audio_og_path]]
    score_all = []

    for i in choices:
        embd_0 = get_embeddings(i[0],classifier).detach().cpu().numpy()
        embd_1 = get_embeddings(i[1],classifier).detach().cpu().numpy()
        similarity = 1-spatial.distance.cosine(embd_0,embd_1)
        score_all.append(similarity)

    return np.asarray(score_all)


class verify_main():

    def __init__(self):
        
        self.metadata_path = [
            '/mnt/d/projects/COVID-datasets/CSS/label/metadata_v2.csv',\
            '/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_v2.csv',
            '/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_v2.csv'
            ]
        self.verifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        
    def start_verify(self):
        
        res_score = []
        
        for dataset in self.metadata_path:
            k = 0
            df = pd.read_csv(dataset) # load metadata
            all_audio_path = df['audio_path'].tolist()
            scores = []
            
            for i in tqdm(all_audio_path):
                if k < 5:
                    try:
                        s = verify_speaker(i,self.verifier)
                        scores.append(s)
                    except:
                        print('Audio not found, skip to the next')
                        pass
                k+=1

            ave_score = np.mean(np.asarray(scores),axis=0)
            print(ave_score)
            res_score.append(ave_score)

        return res_score


# %%
if __name__ == '__main__':

    ASV_result = verify_main().start_verify()
    print(ASV_result)