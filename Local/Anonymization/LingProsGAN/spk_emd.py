import os
import pandas as pd
import json
import torch
import numpy as np
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torchaudio.functional as F
import logging
from tqdm import tqdm
from IMSToucan.TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)

class SpkEncoder():

    def __init__(self, 
                 device='cuda', 
                 enc_source='models/tts/Embedding/embedding_function.pt',
                 model_dir='pretrained_models', 
                 emd_save_dir='exp/covid_data/spk_emd',
                 emd_type:str='style',
                 extract_only=False,
                 ):
        
        self.xvector_encoder = None
        self.ecapa_encoder = None
        self.enc_source = enc_source
        self.device = device
        self.model_dir = model_dir
        self.emd_dir = emd_save_dir
        self.emd_type = emd_type

        if not extract_only:
            if not os.path.exists(self.emd_dir):
                os.mkdir(self.emd_dir)
            logger.info(f"Created a folder named {self.emd_dir} for storing speaker emds.")

        self._init_encoder(emd_type)
    
    def _init_encoder(self,emd_type):
        if emd_type == 'spk':
            self.ecapa_encoder = EncoderClassifier.from_hparams(source=self.enc_source,
                                                                savedir=self.model_dir + '/spkrec-ecapa-voxceleb',
                                                                run_opts={'device': self.device})
            self.xvector_encoder = EncoderClassifier.from_hparams(source=self.enc_source,
                                                            savedir=self.model_dir + '/spkrec-xvect-voxceleb',
                                                            run_opts={'device': self.device})
        elif emd_type == 'style':
            self.audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True,device='cpu')
            self.style_encoder = StyleEmbedding()
            ck = torch.load(self.enc_source)['style_emb_func']
            self.style_encoder.load_state_dict(ck)
            self.style_encoder = self.style_encoder.to(self.device)

    
    def preprocess(self, audio_path):
        signal, sr_og = torchaudio.load(audio_path)
        # handle multi-channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0)

        if sr_og != 16000:
            signal = F.resample(signal,sr_og,new_freq=16000)

        signal  = signal.squeeze()
        signal = signal / torch.max(torch.abs(signal))
        signal = torch.tensor(np.trim_zeros(signal.numpy())) # trim zeros

        if self.emd_type == 'style':
            spec = self.audio_preprocessor.logmelfilterbank(signal, 16000).transpose(0, 1).to(self.device)
            spec_len = torch.LongTensor([len(spec)]).to(self.device)
            return spec, spec_len

        return signal

    def extract_emd_from_audio(self, audio_path, emd_filepath:str=None):
        x = self.preprocess(audio_path)
        if self.emd_type == 'style':
            final_emd = self.style_encoder(x[0].unsqueeze(0),x[1].unsqueeze(1))
        elif self.emd_type == 'spk':
            ecapa_emd = self.ecapa_encoder.encode_batch(x)
            xvect_emd = self.xvector_encoder.encode_batch(x)
            final_emd = torch.cat((xvect_emd, ecapa_emd),dim=-1)
        if emd_filepath is not None:
            torch.save(final_emd, emd_filepath)
        return final_emd
    
    def extract_from_group(self, metadata_file, debug_mode=False):
        print(os. getcwd())
        df = pd.read_csv(metadata_file, delimiter=';')
        df['Embedding_path'] = None
        emd_list = []
        for i,row in tqdm(df.iterrows()):
            folder_path = self.emd_dir
            filename = f'{i}.pt'
            filepath = os.path.join(folder_path, filename)
            logger.info(f"Extracting {row['voice-path-new']}...")
            self.extract_emd_from_audio(row['voice-path-new'],filepath)
            emd_list.append(filepath)
            if ((debug_mode) & (i > 3)):
                return 
        df['Embedding_path'] = emd_list
        df.to_csv(metadata_file,index=False,header=True,sep=';')