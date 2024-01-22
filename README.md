# Overview

This repository contains code used in our [paper](https://arxiv.org/abs/2304.02181) "On the Impact of Voice Anonymization on Speech Diagnostic Applications: a Case Study on COVID-19 Detection".

We explored three voice anonymization methods:

- First one relies on the modification to McAdams coefficients;
- LingGAN (Linguistic-GAN), synthesizes a new speech based on the generated speaker embeddings and pre-extracted phone sequence;
- LingProsGAN (Linguistic-Prosody-GAN), preserves linguistic and prosody information, and had the speaker embedding generator finetuned on pathological speech data.

## Demo audio

We provide three versions of the voice recording, including the original voice, the McAdams-anonymized version, and the GAN-anonymized version. To keep the privacy of participants in the COVID-19 datasets employed in our study, the voice demo is from one of the authors. The speech content remains the same as that used in the COVID sound databases.

[Sentence-Original](https://user-images.githubusercontent.com/48067384/229322700-39c734bc-b40f-4f41-8b3c-09240aa2ca39.mp4)

[Sentence-McAdams-anonymized](https://user-images.githubusercontent.com/48067384/229322711-95a2c666-a71a-41f5-ab80-0452cbb0b09f.mp4)

[Sentence-LingGAN-anonymized
](https://user-images.githubusercontent.com/48067384/229322716-44d8ef45-a2d0-4313-860e-aee18a9a9317.mp4)

## System diagram

<img src="https://user-images.githubusercontent.com/48067384/229264462-fcfe46ee-969d-4e9e-8ecc-d1682e44ee81.png" width="400" height="300">

## Repository structure

```bash
├── Config
├── Graphs
├── Local
│   ├── ASV
│   ├── Anonymization
│   ├── Diagnostics
│   └── FE
└── Results
```

**Config**: contains the (hyper-)parameters used for feature extraction, and model training and evaluation.

**Local**: contains four sub-folders, corresponding to the four blocks of ASD. All scripts can be found in this folder.

- **ASV**: Used for storing pretraiend speaker verification models.
- **Anonymization**: anonymization techniques (systems) used to anonymize speech recordings.
- **Diagnostics**: speech-based diagnostics systems.
- **FE**: feature extraction block, which is comprised of low-level signal processing functions, audio file I/O functions, etc.

**Results**: results of experiments.

**Graphs**: diagrams and so on.

# Pipeline explanation

###### Step-1: Extract features from recordings

Refer to `Local/FE/FE_from_ad.py`. Make sure you have the `librosa`, `openSMILE `and `SRMRpy `installed. If not, please refer to these pages ([librosa](https://librosa.org/doc/main/install.html), [openSMILE](https://audeering.github.io/opensmile-python/); [SRMRpy](https://github.com/jfsantos/SRMRpy)). Modify the feature extraction scripts according to your own metadata files, and how the recordings were saved. The features should be saved in the `Features` folder.

###### Step-2: Anonymize the recordings

Refer to `Local/Anonymization` for the three types of anonymization methods. We provide demos for each type of approach. The GAN ones will need several backbones for the ASR, Prosody, and TTS modules. We provide the backbones in a separate link which will be uploaded soon. In the end, you should have a metadata file (.csv) that stores the paths to the features, paths to the recordings, and label information. If you wish to anonymize single recordings, we also provide examples in the scripts.

###### Step-3: Run evaluation on diagnostic systems

Since there are multiple different settings of anonymizaiton (clean, ignorant, semi-informed, informed, augmented), we stored experimental set-up in separate files, which can be generated using `Config/exp_setup.py`. Change the metadata file paths to your own directory.

Then run `Local/Diagnostics/evaluate.py`, which does model training (SVM, PCA-SVM, BiLSTM) and evaluation (1000* bootstrap; AUC-ROC and UAR) on all anonymization conditions and diagnostics systems.


# Contact

The repo is merged from multiple separate repos, hence might face some minor issues. If you have questions on data acquisition, pipeline set-up, or model details, please contact me by Yi.Zhu@inrs.ca.
