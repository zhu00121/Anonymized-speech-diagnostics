import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from speechbrain.pretrained import EncoderClassifier

from IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor

VALID_VEC_TYPES = {'xvector', 'ecapa', 'ecapa+xvector'}


class DemoSpeakerEmbeddings:

    def __init__(self, vec_type='xvector', device=torch.device('cpu')):
        self.vec_type = vec_type
        assert self.vec_type in VALID_VEC_TYPES, f'Invalid vec_type {self.vec_type}, must be one of {VALID_VEC_TYPES}'
        self.device = device

        self.encoders = []
        if 'ecapa' in self.vec_type:
            self.encoders.append(EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                                savedir='models/speaker_embeddings/spkrec-ecapa-voxceleb',
                                                                run_opts={'device': self.device}))
        if 'xvector' in self.vec_type:
            self.encoders.append(EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb',
                                                                savedir='models/speaker_embeddings/spkrec-xvect-voxceleb',
                                                                run_opts={'device': self.device}))

        self.ap = AudioPreprocessor(input_sr=48000, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024,
                                    cut_silence=False)

    def extract_vector_from_audio(self, wave, sr):
        # adapted from IMSToucan/Preprocessing/AudioPreprocessor
        #norm_wave = self._normalize_wave(wave, sr)
        norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
        norm_wave = torch.tensor(np.trim_zeros(norm_wave.numpy()))

        spk_embs = [encoder.encode_batch(wavs=norm_wave.unsqueeze(0)).squeeze() for encoder in self.encoders]
        if len(spk_embs) == 1:
            return spk_embs[0]
        else:
            return torch.cat(spk_embs, dim=0)

    def _normalize_wave(self, wave, sr):
        # adapted from IMSToucan/Preprocessing/AudioPreprocessor
        wave = torch.tensor(wave)
        print(wave.shape)
        print(wave)
        dur = wave.shape[0] / sr
        wave = wave.squeeze().cpu().numpy()

        # normalize loudness
        meter = pyln.Meter(sr, block_size=min(dur - 0.0001, abs(dur - 0.1)) if dur < 0.4 else 0.4)
        loudness = meter.integrated_loudness(wave)
        loud_normed = pyln.normalize.loudness(wave, loudness, -30.0)
        peak = np.amax(np.abs(loud_normed))
        wave = np.divide(loud_normed, peak)

        wave = torch.Tensor(wave).to(self.device)

        if sr != 16000:
            wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)(wave)

        return wave.cpu()
