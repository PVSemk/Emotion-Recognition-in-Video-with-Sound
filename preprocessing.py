import cv2
import numpy as np
from glob import glob
import os
import json
from tqdm import tqdm
import torch

from align_preprocessing.align_faces import warp_and_crop_face, get_reference_facial_points
from align_preprocessing.retinaface.detector import RetinafaceDetector
from pytorch_face_landmark.models.mobilefacenet import MobileFaceNet
from ABAW2020TNT.face_alignment import draw_mask

detector = RetinafaceDetector()
landmark_detector = MobileFaceNet([112, 112], 136)
landmark_detector.load_state_dict(
    torch.load("/home/pavel/Labs/Diploma/pytorch_face_landmark/checkpoint/mobilefacenet_model_best.pth.tar")['state_dict'])
landmark_detector.eval()
landmark_detector.cuda()

def process(video, metainfo, save_path, output_size=(112, 112)):
    cap = cv2.VideoCapture(video)
    for i in tqdm(range(metainfo["num_frames"])):
        output_file = str(i).zfill(5) + '.jpg'
        output_savepath = os.path.join(save_path, output_file)
        mask_savepath = os.path.join(save_path, "mask", output_file)
        _, frame = cap.read()
        mask_frame = np.zeros(output_size, np.uint8)
        if os.path.exists(output_savepath) and os.path.exists(mask_savepath):
            continue
        try:
            _, facial5points = detector.detect_faces(frame)
            facial5points = np.reshape(facial5points[0], (2, 5))
            default_square = False
            inner_padding_factor = 0.25
            outer_padding = (2, -7)

            # get the reference 5 landmarks position in the crop settings
            reference_5pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding, default_square)

            # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
            dst_img = warp_and_crop_face(frame, facial5points, reference_pts=reference_5pts, crop_size=output_size)
            landmark_input = dst_img.copy() / 255
            landmark_input = torch.from_numpy(landmark_input.transpose((2, 0, 1))).unsqueeze(0).float().cuda()
            landmark = landmark_detector(landmark_input)[0].cpu().data.numpy()
            landmark = landmark.reshape((-1, 2))
            landmark = landmark * dst_img.shape[0:2]
            draw_mask(landmark, mask_frame)

        except IndexError:
            dst_img = np.zeros((112, 112, 3), np.uint8)

        cv2.imwrite(output_savepath, dst_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(mask_savepath, mask_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # img = cv.resize(raw, (224, 224))
        # cv.imwrite('images/{}_img.jpg'.format(i), img)


def preprocess_img(img, output_size=(112, 112)):
    global detector, landmark_detector
    mask_img = np.zeros(output_size, np.uint8)
    try:
        _, facial5points = detector.detect_faces(img)
        facial5points = np.reshape(facial5points[0], (2, 5))
        default_square = False
        inner_padding_factor = 0.25
        outer_padding = (2, -7)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)

        # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
        dst_img = warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
        landmark_input = dst_img.copy() / 255
        landmark_input = torch.from_numpy(landmark_input.transpose((2, 0, 1))).unsqueeze(0).float().cuda()
        landmark = landmark_detector(landmark_input)[0].cpu().data.numpy()
        landmark = landmark.reshape((-1, 2))
        landmark = landmark * dst_img.shape[0:2]
        draw_mask(landmark, mask_img)

    except IndexError:
        dst_img = np.zeros((112, 112, 3), np.uint8)

    return dst_img, mask_img


if __name__ == "__main__":
    detector = RetinafaceDetector()
    landmark_detector = MobileFaceNet([112, 112], 136)
    landmark_detector.load_state_dict(torch.load("pytorch_face_landmark/checkpoint/mobilefacenet_model_best.pth.tar")['state_dict'])
    landmark_detector.eval()
    landmark_detector.cuda()
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ABAW2020TNT", "dataset", "tnt_prepared_data", "subset")
    videos = glob(os.path.join(video_path, "*.mp4"))
    videos.extend(glob(os.path.join(video_path, "*.avi")))
    for video in sorted(videos):
        print(f"Processing: {video.split('/')[-1]}")
        save_path = os.path.join(video_path, "..", "custom_preprocessing", os.path.splitext(os.path.basename(video))[0])
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "mask"), exist_ok=True)
        with open(video + "meta.json", "r") as jfile:
            metainfo = json.load(jfile)
        process(video, metainfo, save_path)


