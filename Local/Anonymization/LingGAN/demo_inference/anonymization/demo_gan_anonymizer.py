from pathlib import Path
import torch
import numpy as np
from scipy.spatial.distance import cosine
import json

from .demo_speaker_embeddings import DemoSpeakerEmbeddings


class DemoGANAnonymizer:

    def __init__(self, vec_type='xvector', device=None, sim_threshold=0.7):
        self.vec_type = vec_type
        self.device = device

        self.sim_threshold = sim_threshold

        self.embedding_extractor = DemoSpeakerEmbeddings(vec_type=self.vec_type, device=self.device)

        self.model_dir = None
        self.vectors_file = None
        self.gan_vectors = None
        self.unused_indices = None

    def load_parameters(self, model_dir: Path):
        self.model_dir = model_dir
        with open(model_dir / 'settings.json') as f:
            settings = json.load(f)
        self.vec_type = settings.get('vec_type', self.vec_type)
        self.vectors_file = settings.get('vectors_file', self.vectors_file)

        self.gan_vectors = torch.load(model_dir / self.vectors_file, map_location=self.device)
        self.unused_indices = self.load_unused_indices()

    def load_unused_indices(self):
        return torch.load(self.model_dir / f'unused_indices_{self.vectors_file}', map_location='cpu')

    def anonymize_embedding(self, audio, sr):
        speaker_embedding = self.embedding_extractor.extract_vector_from_audio(wave=audio, sr=sr)
        anon_vec = self._select_gan_vector(spk_vec=speaker_embedding)
        return anon_vec

    def _select_gan_vector(self, spk_vec):
        i = 0
        limit = 20
        while i < limit:
            idx = np.random.choice(self.unused_indices)
            anon_vec = self.gan_vectors[idx]
            sim = 1 - cosine(spk_vec.cpu().numpy(), anon_vec.cpu().numpy())
            if sim < self.sim_threshold:
                break
            i += 1
        self.unused_indices = self.unused_indices[self.unused_indices != idx]
        if len(self.unused_indices) == 0:
            self.unused_indices = self.load_unused_indices()
        return anon_vec