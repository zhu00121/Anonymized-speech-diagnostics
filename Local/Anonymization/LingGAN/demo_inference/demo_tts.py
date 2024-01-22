from IMSToucan.InferenceInterfaces.AnonFastSpeech2 import AnonFastSpeech2


class DemoTTS:

    def __init__(self, model_paths, device):
        self.device = device
        fastspeech_path = model_paths / 'FastSpeech2_Multi' / 'trained_on_ground_truth_phonemes.pt'
        hifigan_path = model_paths / 'HiFiGAN_combined' / 'best.pt'
        self.model = AnonFastSpeech2(device=self.device, path_to_hifigan_model=hifigan_path,
                                     path_to_fastspeech_model=fastspeech_path)

    def read_text(self, transcription, speaker_embedding, text_is_phonemes=False):
        self.model.default_utterance_embedding = speaker_embedding.to(self.device)
        wav = self.model(text=transcription, text_is_phonemes=text_is_phonemes)
        return wav
