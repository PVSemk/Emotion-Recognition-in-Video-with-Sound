# Emotion-Recognition-in-Video-with-Sound

This repository contains preprocessing scripts for AffWild2 dataset and PyTorch-Lightning implementation of CNN, CNN+Audio and RNN models to solve the Valence-Arousal emotion estimation issue.

## Preparation

Install requirements with `pip install -r requirements.txt`

## Preprocessing
We provide 2 ways to preprocess the data: with OpenFace lib and with face_alignment lib

Preprocessing scripts are under `Diploma/preprocessing`

### OpenFace

To crop and align faces with OpenFace lib do:
1. Clone [the repository](https://github.com/TadasBaltrusaitis/OpenFace)
2. Make sure you have docker and docker-compose installed
3. Provide environment variable DATA_MOUNT - path to folder with AffWild2 video files
4. Launch `python offline_preprocessing.py`, cropped and aligned faces will be stored under `$DATA_MOUNT/aligned_faces/video_name`

### Face_alignment Lib

Use `python face_alignment_base_preprocessing.py -v /path/to/videos -s /save/path -a /path/to/annotations` to crop & align faces

See `python face_alignment_base_preprocessing.py --help` for more details and arguments.

### Audio
1. Extract audio from videos and store it in `.wav` format with `./extract_audio.sh /path/to/videos path/to/save`
2. Use `python audio_preprocessing.py -i /path/to/wavs -o /path/to/save -v /path/to/videos`

See `python audio_preprocessing.py --help` for more details and arguments.
   
## Training
The models are developed with PyTorch and trained with PyTorch-Lightning. Scripts are stored under `Diploma/train`

To train the model setup the config (see `config_affwild.yaml` for example) and launch `python train_pl --cfg config_affwild.yaml` 

To change the augmentations, see `transforms.py`

To use audio model, provide path to preprocessed audio with `audio_path` key in config

To use RNN model, declare `seq_len` and set `rnn: True` in config. Note that RNN+audio is not supported and wasn't tested

## Testing
To test the model, specify in config:
- `test: True`
- `checkpoint: /path/to/pretrained/weights`