from pathlib import Path
import numpy as np
import torch
import json
from sklearn.metrics.pairwise import cosine_distances

from .plda_model import PLDAModel
from .demo_speaker_embeddings import DemoSpeakerEmbeddings


class DemoPoolAnonymizer:

    def __init__(self, vec_type='xvector', N=200, N_star=100, distance='plda', proximity='farthest', device=None):
        # Pool anonymization method based on the primary baseline of the Voice Privacy Challenge 2020.
        # Given a speaker vector, the N most distant vectors in an external speaker pool are extracted,
        # and an average of a random subset of N_star vectors is computed and taken as new speaker vector.
        # Default distance measure is PLDA.
        self.vec_type = vec_type
        self.device = device

        self.N = N  # number of most distant vectors to consider
        self.N_star = N_star  # number of vectors to include in averaged vector
        self.distance = distance  # distance measure, either 'plda' or 'cosine'
        self.proximity = proximity  # proximity method, either 'farthest' (distant vectors), 'nearest', or 'closest'

        self.embedding_extractor = DemoSpeakerEmbeddings(vec_type=self.vec_type, device=self.device)

        self.pool_embeddings = None
        self.plda = None

    def load_parameters(self, model_dir: Path):
        self._load_settings(model_dir / 'settings.json')
        self.pool_embeddings = torch.load(model_dir / 'pool_embeddings' / f'speaker_vectors.pt',
                                          map_location=self.device)
        if self.distance == 'plda':
            self.plda = PLDAModel(train_embeddings=None, results_path=model_dir)

    def anonymize_embedding(self, audio, sr):
        speaker_embedding = self.embedding_extractor.extract_vector_from_audio(wave=audio, sr=sr)

        distances = self._compute_distances(vectors_a=self.pool_embeddings,
                                            vectors_b=speaker_embedding.unsqueeze(0)).squeeze()

        candidates = self._get_pool_candidates(distances)
        selected_anon_pool = np.random.choice(candidates, self.N_star, replace=False)
        anon_vec = torch.mean(self.pool_embeddings[selected_anon_pool], dim=0)

        return anon_vec

    def _compute_distances(self, vectors_a, vectors_b):
        if self.distance == 'plda':
            return 1 - self.plda.compute_distance(enrollment_vectors=vectors_a, trial_vectors=vectors_b)
        elif self.distance == 'cosine':
            return cosine_distances(X=vectors_a.cpu(), Y=vectors_b.cpu())
        else:
            return []

    def _get_pool_candidates(self, distances):
        if self.proximity == 'farthest':
            return np.argpartition(distances, -self.N)[-self.N:]
        elif self.proximity == 'nearest':
            return np.argpartition(distances, self.N)[:self.N]
        elif self.proximity == 'center':
            sorted_distances = np.sort(distances)
            return sorted_distances[len(sorted_distances)//2:(len(sorted_distances)//2)+self.N]

    def _load_settings(self, filename):
        with open(filename, 'r') as f:
            settings = json.load(f)

        self.N = settings['N'] if 'N' in settings else self.N
        self.N_star = settings['N*'] if 'N*' in settings else self.N_star
        self.distance = settings['distance'] if 'distance' in settings else self.distance
        self.proximity = settings['proximity'] if 'proximity' in settings else self.proximity
        self.vec_type = settings['vec_type'] if 'vec_type' in settings else self.vec_type
