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

## Repository structure [TBD]

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

**Config**: contains the (hyper-)parameters used for feature extraction, and model training and evaluation. `<br />`

**Local**: contains four sub-folders, corresponding to the four blocks of ASD. All scripts can be found in this folder.  `<br />`

- **ASV**: Used for storing pretraiend speaker verification models. `<br />`
- **Anonymization**: anonymization techniques (systems) used to anonymize speech recordings. `<br />`
- **Diagnostics**: speech-based diagnostics systems. `<br />`
- **FE**: feature extraction block, which is comprised of low-level signal processing functions, audio file I/O functions, etc. `<br />`

**Results**: results of experiments. `<br />`
**Graphs**: diagrams and so on.
