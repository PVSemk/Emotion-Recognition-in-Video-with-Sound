from openface_utils.video_processor import Video_Processor
import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from openface_utils.resnet50_extractor import Resnet50_Extractor

VIDEO_EXT = ['.mp4', '.avi']
class Preprocessor(object):
    def __init__(self,
                 save_size=112, nomask=True, grey=False, quiet=True,
                 tracked_vid=False, noface_save=False,
                 ):
        self.video_processor = Video_Processor(save_size, nomask, grey, quiet,
                                               tracked_vid, noface_save)

    def run(self):
        call = "docker-compose -f openface/docker-compose.yml up -d openface"
        os.system(call)
        data_mount_path = os.getenv("DATA_MOUNT")
        if not data_mount_path:
            raise EnvironmentError("DATA_MOUNT variable wasn't found")
        for root, dirs, files in os.walk(data_mount_path):
            for file in tqdm(files):
                if os.path.splitext(file)[1] in VIDEO_EXT:
                    input_video = os.path.join(root, file)
                    video_name = os.path.basename(input_video).split('.')[0]
                    output_dir = os.path.join(root, "aligned_faces", video_name)
                    self.video_processor.process(input_video, output_dir)
        call = "docker-compose -f openface/docker-compose.yml stop"
        os.system(call)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
