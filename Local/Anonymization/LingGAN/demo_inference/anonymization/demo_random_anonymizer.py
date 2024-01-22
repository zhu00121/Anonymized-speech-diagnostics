import json
import torch
import numpy as np

from .demo_speaker_embeddings import DemoSpeakerEmbeddings


class DemoRandomAnonymizer:

    def __init__(self, device, vec_type='xvector', in_scale=False):
        self.device = device
        self.vec_type = vec_type
        self.in_scale = in_scale
        self.dim_ranges = None
        self.embedding_extractor = DemoSpeakerEmbeddings(vec_type=self.vec_type, device=self.device)

    def load_parameters(self, model_dir):
        with open(model_dir / 'settings.json') as f:
            settings = json.load(f)
        self.vec_type = settings['vec_type'] if 'vec_type' in settings else self.vec_type
        self.in_scale = settings.get('in_scale', self.in_scale)

        if self.in_scale:
            with open(model_dir / 'stats_per_dim.json') as f:
                dim_ranges = json.load(f)
                self.dim_ranges = [(v['min'], v['max']) for k, v in sorted(dim_ranges.items(), key=lambda x: int(x[0]))]

    def anonymize_embedding(self, audio, sr):
        speaker_embedding = torch.tensor(self.embedding_extractor.extract_vector_from_audio(wave=audio, sr=sr))

        if self.dim_ranges:
            anon_vec = torch.tensor([np.random.uniform(*dim_range) for dim_range in self.dim_ranges]).to(self.device)
        else:
            mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-40, 40).to(self.device)
            anon_vec = speaker_embedding * mask

        return anon_vec
