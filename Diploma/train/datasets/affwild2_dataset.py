import os

import cv2
from tqdm import tqdm
from torch.utils.data import Dataset


class AffWildVADataset(Dataset):
    def __init__(self, data_path, label_path, transform, mode='train'):
        self.label_path = label_path
        self.data_path = data_path
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
            video_folder = os.path.join(self.data_path, video_folder)
            assert os.path.exists(video_folder)

            with open(label_file, 'r') as f:
                labels = f.readlines()[1:]
            labels = [l.split() for l in labels]

            for frame_idx, label in enumerate(labels, start=1):
                # get every 30 frames (1 FPS)
                if (frame_idx-1) % skip_frame != 0:
                    continue

                image_name = f'{str(frame_idx).zfill(5)}.jpg'
                image_path = os.path.join(video_folder, image_name)
                if not os.path.exists(image_path):
                    continue

                label = label[0]
                if "-5" in label:
                    continue
                valence, arousal = label.split(',')
                valence, arousal = float(valence), float(arousal)

                self.data[counter] = {
                    'image_path': image_path,
                    'targets': {'valence': valence,
                                'arousal': arousal
                                }
                }
                counter += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image = cv2.imread(item["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(**{"image": image})
        image = augmented["image"]

        return {'image': image, "targets": item["targets"]}