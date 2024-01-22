from espnet2.bin.asr_inference import Speech2Text
import resampy
from espnet_model_zoo.downloader import ModelDownloader




class DemoASR:

    def __init__(self, model_path, device):
        model_file = 'asr_improved_tts-phn_en.zip'

        d = ModelDownloader()
        
        self.device = device
        self.speech2text = Speech2Text(
            **d.download_and_unpack(str(model_path / model_file)),
            device=self.device,
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=0.4,
            beam_size=15,
            batch_size=1,
            nbest=1
        )

    def recognize_speech(self, audio, sr):
        if len(audio.shape) == 2:
            audio = audio.T[0]
        speech = resampy.resample(audio, sr, 16000)
        # speech = librosa.resample(audio,sr,16000)
        nbests = self.speech2text(speech)
        text, *_ = nbests[0]
        return nbests
