"""
This is the modified LingProsGAN from 
https://github.com/DigitalPhonetics/speaker-anonymization/tree/main

The generator is finetuned on COVID data, while we are not able to provide the backbone due to data confidentiality reasons,
we provide the training script in 'train.py'.

Usage:
    m = LingProsGAN()
    df, ct = m.anonymize_dataset(metadata_path='YOUR_METADATA.csv',anonymize='pros')
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_data import *
from anonymization.WGAN.init_wgan import create_wgan
import logging
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import numpy as np
import yaml
from inference.asr import InferenceASR
from IMSToucan.UtteranceCloner import UtteranceCloner
from anonymization.WGAN.init_wgan import create_wgan
from exp.spk_emd import SpkEncoder
from scipy.spatial.distance import cosine
import timeit
import util
from tqdm import tqdm


class LingProsGAN(object):

    def __init__(self,
                 asr_model_name:str='asr_branchformer_tts-phn_en.zip',
                 model_dir='models',
                 path_to_hifigan_model='models/tts/HiFiGAN_combined/best.pt', 
                 path_to_fastspeech_model='models/tts/FastSpeech2_Multi/prosody_cloning.pt',
                 path_to_aligner_model='models/tts/Aligner/aligner.pt', 
                 path_to_embed_model='models/tts/Embedding/embedding_function.pt',
                 path_to_gan_hyperparameter='pretrained_models/gan/style-embed_wgan.pt',
                 path_to_gan_ckpt='pretrained_models/gan/gan.pt',
                 device='cuda'
                 ):
        
        # we load the asr on cpu and the rest on GPU
        self.asr_model = InferenceASR(
            model_name=asr_model_name,
            data_dir=None,
            model_dir=model_dir,
            device='cpu',
            results_dir=None,
        )

        self.tts_model = UtteranceCloner(
            path_to_hifigan_model=path_to_hifigan_model,
            path_to_fastspeech_model=path_to_fastspeech_model,
            path_to_aligner_model=path_to_aligner_model,
            path_to_embed_model=path_to_embed_model,
            device=device,
        )

        self.spk_enc = SpkEncoder(extract_only=True)
        
        self.device=device
        self._init_gan(path_to_gan_hyperparameter,path_to_gan_ckpt)


    def _init_gan(self,path_to_gan_hyperparameter,path_to_gan_ckpt):
        gan_param = torch.load(path_to_gan_hyperparameter, map_location=self.device)['model_parameters']
        if not hasattr(gan_param, 'opt_iterations'): gan_param['opt_iterations'] = 1
        self.spk_gan = create_wgan(parameters=gan_param, device=self.device)
        self.spk_gan.G.load_state_dict(torch.load(path_to_gan_ckpt)) # load the finetuned generator
    
    def look_for_match(self, emd_og, emd_gen, threshold=0.7):
        sim_list = []
        for emd in emd_gen:
            sim = cosine(emd_og.flatten(),emd.flatten())
            sim_list.append(sim)
            if sim >= threshold:
                return emd # if found the emd satisfying the threshold, return it
        print('Did not find any that satisfied the threshold; chose the most similar one')
        return emd_gen(np.argmax(sim_list)) # else return the one with the highest similarity

    def anonymize(self,path_to_ad,num_to_gen=20,similarity=0.7,path_new_ad=None):
        # get original spk emd
        emd_og = self.spk_enc.extract_emd_from_audio(path_to_ad,emd_filepath=None).detach().cpu().numpy()
        # look for the generated emd that satisfies the cosine similarity constraint
        emd_gen = self.spk_gan.sample_generator(num_samples=num_to_gen, nograd=True, return_intermediate=False).detach().cpu().numpy()
        emd_new = torch.Tensor(self.look_for_match(emd_og, emd_gen, similarity))
        # get the text
        text = self. asr_model.recognize_speech(path_to_ad)
        # synthesize with prosody cloned internally
        utt_new = self.tts_model.clone_utterance(
                    path_to_reference_audio=path_to_ad,
                    reference_transcription=text,
                    clone_speaker_identity=False,
                    speaker_embedding=emd_new.squeeze(),
                    lang="en",
                    input_is_phones='-ph',
                    random_offset_lower=None,
                    random_offset_higher=None
                    )

        # save new audio
        if path_new_ad is not None:
            torchaudio.save(path_new_ad,utt_new.view(1,-1).detach().cpu(),sample_rate=48000)
        return utt_new

    def anonymize_dataset(self, metadata_path:str, anonymize:str, start_from=None, ano_metadata_path:str=None):

        assert os.path.exists(metadata_path), "Required metadata file is not in this location"

        if ano_metadata_path is None:
            ano_metadata_path = os.path.join(os.path.dirname(metadata_path),'metadata_v2_%s.csv'%(anonymize))

        df_md = pd.read_csv(metadata_path) # load metadata
        assert 'audio_path' in df_md.columns.values, "Required info missing in metadata"

        all_audio_path = df_md['audio_path'].tolist()
        ano_audio_path = []
        error_files=[]
        compute_time = []

        print('Start anonymization | anonymizer: %s ...'%(anonymize))

        for ad_path in tqdm(all_audio_path):
            head, tail = os.path.split(ad_path)
            sample_id = util.remove_suffix(tail,'.wav')
            sample_id = util.remove_suffix(sample_id,'.flac')
            new_name = sample_id+'_%s.wav'%(anonymize)
            new_path = os.path.join(head,new_name) # anonymized audio path
            ano_audio_path.append(new_path)
            # anonymize audio
            try:
                starttime = timeit.default_timer()
                self.anonymize(path_to_ad=ad_path, path_new_ad=new_path)
                t = timeit.default_timer() - starttime
                compute_time.append(t)
            except:
                print(f'Error occurred in {ad_path}')
                error_files.append(ad_path)
                pass

        ave_computetime = np.mean(compute_time)
        std_computetime = np.std(compute_time)

        print('Anonymization completed | %s'%(anonymize))
        print('Average time taken for anonymize one audio file is %f'%(ave_computetime))
        print('Std time taken for anonymize one audio file is %f'%(std_computetime))
        print(error_files)

        # Save feature path
        df_md_ano = df_md
        df_md_ano['audio_path'] = ano_audio_path
        df_md_ano.to_csv(ano_metadata_path)

        return df_md_ano, [ave_computetime, std_computetime]