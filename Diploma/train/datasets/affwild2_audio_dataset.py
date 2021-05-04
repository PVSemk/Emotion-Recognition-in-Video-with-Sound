import os

import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np

import librosa.display
import matplotlib.pyplot as plt
class AffWildVADatasetWithAudio(Dataset):
    def __init__(self, data_path, label_path, audio_path, transform, mode='train'):
        self.label_path = label_path
        self.data_path = data_path
        self.audio_path = audio_path
        self.mode = mode
        self.data = {}

        self.get_labels()
        self.get_data(skip_frame=5)

        self.transform = transform

    def get_labels(self):
        if self.mode == 'train':
            task_set_path = "Train_Set"
        elif self.mode == 'val':
            task_set_path = "Validation_Set"
        else:
            raise NotImplementedError(f'There is no mode {self.mode}, it can be changed with [\'train\', \'val\']')

        self.label_path = os.path.join(self.label_path, task_set_path)
        self.label_files = [os.path.join(self.label_path, f) for f in os.listdir(self.label_path) if f.endswith('.txt')]
        self.label_files = self.label_files

    def get_data(self, skip_frame=30):
        videos = [os.path.basename(f)[:-4] for f in self.label_files]
        assert len(videos) == len(self.label_files)

        counter = 0

        for i in tqdm(range(len(videos))):
            video_folder = videos[i]
            label_file = self.label_files[i]

            assert os.path.basename(label_file)[:-4] == video_folder
            # task_set_path = ""
            # if self.mode == 'train':
            #     task_set_path = 'Train_Set'
            # elif self.mode == 'val':
            #     task_set_path = 'Validation_Set'
            # video_folder = os.path.join(self.data_path, task_set_path, video_folder)
            audio_folder = os.path.join(self.audio_path, video_folder)
            video_folder = os.path.join(self.data_path, video_folder)
            if not (os.path.exists(video_folder) and os.path.exists(audio_folder)):
                continue

            with open(label_file, 'r') as f:
                labels = f.readlines()[1:]
            labels = [l.split() for l in labels]

            for frame_idx, label in enumerate(labels, start=1):
                # get every 30 frames (1 FPS)
                if (frame_idx-1) % skip_frame != 0:
                    continue

                image_name = f'{str(frame_idx).zfill(5)}.jpg'
                audio_name = f'{str(frame_idx - 1).zfill(5)}.npy'
                image_path = os.path.join(video_folder, image_name)
                audio_frame_path = os.path.join(audio_folder, audio_name)
                if not os.path.exists(image_path) or not os.path.exists(audio_frame_path):
                    continue

                label = label[0]
                if "-5" in label:
                    continue
                valence, arousal = label.split(',')
                valence, arousal = float(valence), float(arousal)

                self.data[counter] = {
                    'image_path': image_path,
                    'audio_path': audio_frame_path,
                    'targets': {'valence': valence,
                                'arousal': arousal
                                }
                }
                counter += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        audio = np.load(item["audio_path"])
        # librosa.display.specshow(audio, x_axis='time',
        #                          y_axis='mel', sr=48000)
        # plt.show()
        # Calculated mean for audio is -42.4245, std 22.5223
        audio = ((torch.from_numpy(audio.transpose()) + 42.4245) / 22.5223).unsqueeze(0)
        image = cv2.imread(item["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(**{"image": image})
        image = augmented["image"]

        return {'image': image, "audio": audio, "targets": item["targets"]}