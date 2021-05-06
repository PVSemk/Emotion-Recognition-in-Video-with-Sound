import os

import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


class AffWildVADatasetRNN(Dataset):
    def __init__(self, data_path, label_path, transform, seq_len, mode='train'):
        self.label_path = label_path
        self.data_path = data_path
        self.mode = mode
        self.data = {}
        self.seq_len = seq_len
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
            video_folder = os.path.join(self.data_path, video_folder)
            assert os.path.exists(video_folder)

            with open(label_file, 'r') as f:
                labels = f.readlines()[1:]
            labels = [l.split() for l in labels]
            labels_dict = {}

            for frame_idx, label in enumerate(labels, start=1):
                # get every 5 frames (6 FPS)
                if (frame_idx-1) % skip_frame != 0:
                    continue
                image_name = f'{str(frame_idx).zfill(5)}.jpg'
                image_path = os.path.join(video_folder, image_name)
                if not os.path.exists(image_path):
                    continue
                label = label[0]
                valence, arousal = label.split(',')
                valence, arousal = float(valence), float(arousal)
                labels_dict[frame_idx] = valence, arousal

            for frame_idx, label in labels_dict.items():
                seq_idx = [frame_idx - i * skip_frame for i in range(1, self.seq_len // 2)]
                seq_idx.extend([frame_idx + i * skip_frame for i in range(1, self.seq_len // 2 + 1,)])
                seq_idx.append(frame_idx)
                seq_idx.sort()
                anchor_frame_idx = seq_idx.index(frame_idx)
                for i in range(anchor_frame_idx, -1, -1):
                    if seq_idx[i] not in labels_dict.keys():
                        seq_idx[i] = seq_idx[i + 1]
                for i in range(anchor_frame_idx, self.seq_len):
                    if seq_idx[i] not in labels_dict.keys():
                        seq_idx[i] = seq_idx[i - 1]
                seq_names = [os.path.join(video_folder, f'{str(idx).zfill(5)}.jpg') for idx in seq_idx]
                valence = [labels_dict[ix][0] for ix in seq_idx]
                arousal = [labels_dict[ix][1] for ix in seq_idx]
                self.data[counter] = {
                    'images': seq_names,
                    'targets': {'valence': valence,
                                'arousal': arousal
                                }
                }
                counter += 1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        seq = []
        for path in item["images"]:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.transform(**{"image": image})
            image = augmented["image"]
            seq.append(image)
        seq = torch.stack(seq)
        item["targets"]["arousal"] = torch.Tensor(item["targets"]["arousal"])
        item["targets"]["valence"] = torch.Tensor(item["targets"]["valence"])

        return {'image': seq, "targets": item["targets"]}