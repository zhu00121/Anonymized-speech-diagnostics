"""
The Ling-GAN is an off-the-shelf anonymizer from 
https://github.com/DigitalPhonetics/speaker-anonymization/tree/gan_embeddings

Please refer to the 'Ling-GAN' function and change the input and output audio path to your own directory;
Note that Ling-GAN keeps only the phoneme sequence hence loses the speaker and prosody information.

Usage:
    model = VPInterface()
    LingGAN_anonymize(input_audio_path, output_audio_path, model)
"""

import os
import sys
import gradio as gr
import numpy as np
import torch
from pathlib import Path
from scipy.io import wavfile
import librosa
import warnings

# os.system("pip uninstall -y gradio")
# os.system("pip install gradio==3.2")

from demo_inference.demo_tts import DemoTTS
from demo_inference.demo_asr import DemoASR
from demo_inference.demo_anonymization import DemoAnonymizer


def pcm2float(sig, dtype='float32'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


class VPInterface:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.path_to_tts_models = Path('/mnt/d/projects/speaker-anonymization-gan/models', 'tts')
        self.path_to_asr_model = Path('/mnt/d/projects/speaker-anonymization-gan/models', 'asr')
        self.path_to_anon_model = Path('/mnt/d/projects/speaker-anonymization-gan/models', 'anonymization')

        self.synthesis_model = DemoTTS(model_paths=self.path_to_tts_models, device=self.device)
        self.asr_model = DemoASR(model_path=self.path_to_asr_model, device=self.device)
        self.anon_model = DemoAnonymizer(model_path=self.path_to_anon_model, model_tag='gan', device=self.device)

    def read(self, recording, anon_model_tag):
        sr, audio = recording
        # audio = pcm2float(audio)

        self._check_models(anon_model_tag)

        text_is_phonemes = True
        text = self.asr_model.recognize_speech(audio, sr)
        speaker_embedding = self.anon_model.anonymize_embedding(audio, sr)
        syn_audio = self.synthesis_model.read_text(transcription=text, speaker_embedding=speaker_embedding,
                                                   text_is_phonemes=text_is_phonemes)

        return 48000, float2pcm(syn_audio.cpu().numpy())
        # return 48000, syn_audio

    def _check_models(self, anon_model_tag):
        if anon_model_tag != self.anon_model.model_tag:
            self.anon_model = DemoAnonymizer(model_path=self.path_to_anon_model, model_tag=anon_model_tag,
                                             device=self.device)

# %% test with sample audio
def soundDataToInt(SD):
    "Converts librosa's float-based representation to int, given a numpy array SD"
    return [ int(s*32768) for s in SD]

def soundIntToData(SD):
    return np.asarray([ s/32768 for s in SD],dtype=np.float32)

def LingGAN_anonymize(input_path,output_path,model):

    warnings.filterwarnings("ignore")
    
    # load audio
    sig, fs = librosa.load(input_path,sr=16000)
    # sig = soundDataToInt(sig)
    if not (np.isfinite(sig).all()):
        warnings.warn("NaN or infinity in signal, substitute with 0")
        sig = np.zeros((fs,))

    # anonymize the audio
    _, sig_rec = model.read((fs,sig),'gan') # sr is 48000 here, need downsampling

    # resample
    sig_rec = soundIntToData(sig_rec)
    sig_rec = librosa.resample(sig_rec,orig_sr=48000,target_sr=16000)

    # rescale
    sig_rec = sig_rec/np.max(np.abs(sig_rec))

    # save to output path
    if (np.isnan(sig_rec).any()): 
        warnings.warn("NaN in anonymized audio. Original audio is saved instead.")
        assert not (np.isnan(sig).any())
        wavfile.write(output_path, fs, np.float32(sig))
    elif not (np.isnan(sig_rec).any()):
        wavfile.write(output_path, fs, np.float32(sig_rec)) 