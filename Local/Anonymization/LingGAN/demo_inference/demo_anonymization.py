import json
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale, StandardScaler

from anonymization import DemoPoolAnonymizer, DemoRandomAnonymizer, DemoGANAnonymizer

TAGS_TO_MODELS = {
    'pool': 'pool_minmax_ecapa+xvector',
    'random': 'random_in-scale_ecapa+xvector',
    'gan': 'gan'
}

ANON_MODELS = {
    'pool': DemoPoolAnonymizer,
    'random': DemoRandomAnonymizer,
    'gan': DemoGANAnonymizer
}


class DemoAnonymizer:

    def __init__(self, model_path, model_tag, device):
        self.device = device
        self.scaling = None
        self.std_scaler = None
        self.model_tag = model_tag

        self.dim_ranges = self._load_dim_ranges(model_path / TAGS_TO_MODELS[model_tag])
        self.anonymizer = self._load_anonymizer(model_path / TAGS_TO_MODELS[model_tag])

    def anonymize_embedding(self, audio, sr):

        anon_embedding = self.anonymizer.anonymize_embedding(audio, sr)
        if self.dim_ranges:
            anon_embedding = self._scale_embedding(anon_embedding)
        return anon_embedding

    def _load_dim_ranges(self, model_dir):
        if (model_dir / 'stats_per_dim.json').exists():
            with open(model_dir / 'stats_per_dim.json') as f:
                dim_ranges = json.load(f)
                return [(v['min'], v['max']) for k, v in sorted(dim_ranges.items(), key=lambda x: int(x[0]))]

    def _load_anonymizer(self, model_dir):
        model_name = model_dir.name.lower()

        if 'pool' in model_name:
            model_type = 'pool'
        elif 'gan' in model_name:
            model_type = 'gan'
        else:
            model_type = 'random'

        print(f'Model type of anonymizer: {model_type}')

        model = ANON_MODELS[model_type](device=self.device, vec_type='ecapa+xvector')
        model.load_parameters(model_dir)

        if 'minmax' in model_name:
            self.scaling = 'std'
            self.std_scaler = StandardScaler()
            self.std_scaler.fit(model.pool_embeddings.cpu().numpy())

        return model

    def _scale_embedding(self, vector):
        if self.scaling == 'minmax':
            vector = vector.cpu().numpy()
            scaled_dims = []
            for i in range(len(self.dim_ranges)):
                scaled_dims.append(minmax_scale(np.array([vector[i]]), self.dim_ranges[i])[0])

            vector = torch.tensor(scaled_dims).to(self.device)
        elif self.scaling == 'std':
            vector = vector.unsqueeze(0).cpu().numpy()
            vector = torch.tensor(self.std_scaler.transform(vector)[0])

        return vector
