import os
import cv2
import time
import json

import torch
import numpy as np
import face_alignment
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from joblib import Parallel, delayed
from face_alignment_utils.face_aligner import get_centers, get_eyes
from face_alignment_utils.utils import warp_and_crop_face, get_reference_facial_points

from torch.multiprocessing import set_start_method, Pool, Queue
try:
     set_start_method('spawn')
except RuntimeError:
    pass



def filter_na(faces, names):
    _faces, _names = [], []
    for face, name in zip(faces, names):
        if face is not None:
            _faces.append(face)
            _names.append(name)
        # else:
        #     print(f"No face detected in the {name} frame")

    return _faces, _names


def save_faces(faces, names, save_dir='../save_dir'):
    names = [os.path.join(save_dir, name) for name in names]
    faces, names = filter_na(faces, names)
    for i, batch in enumerate(faces):
        # for face in batch:
        cv2.imwrite(names[i], batch)


def get_one_face(predictions, side):
    for i, pred in enumerate(predictions):
        if pred is not None and len(np.array(pred).shape) > 2:
            if side == 'right':
                idx = pred[:, 33, 0].argmax()
                predictions[i] = pred[idx][np.newaxis, ...]
            elif side == 'left':
                idx = pred[:, 33, 0].argmin()
                predictions[i] = pred[idx][np.newaxis, ...]
    return predictions


# def align_face(image, landmarks_68):
#     eyes = get_eyes(landmarks_68)
#     centers = get_centers(eyes)

def batch_wrapper(image, landmarks):
    batch_num = len(landmarks)
    output = []
    for i in range(batch_num):
        if len(landmarks[i]) > 0:
            landmarks_img = np.array(landmarks[i])
            if len(landmarks_img.shape) == 2:
                landmarks_img = landmarks_img[np.newaxis, ...]
            eyes_landmarks = get_eyes(landmarks_img)
            left, right = get_centers(eyes_landmarks)
            facial_points_5 = np.array([left[0], right[0], landmarks_img[0, 33], landmarks_img[0, 48], landmarks_img[0, 54]])
            cropped_face = warp_and_crop_face(image[i], facial_points_5.T,
                                              reference_pts=reference_5pts,
                                              crop_size=(112, 112))
            output.append(cropped_face)
        else:
            output.append(None)

    return output



def crop(image, x_start, y_start, x_end, y_end):
    image = image[y_start:y_end, x_start:x_end,]
    return image


def resize(image, window_height=500):
    if image.shape[0] < window_height:
        return image
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    window_width = window_height/aspect_ratio
    image = cv2.resize(image, (int(window_height), int(window_width)))
    return image


def norm_landmarks(landmarks, w, h):
    for landmark in landmarks:
        if len(landmark) > 0:
            landmark[..., 0] = landmark[..., 0]/w
            landmark[..., 1] = landmark[..., 1]/h
    return landmarks


def denorm_landmarks(landmarks, w, h):
    for landmark in landmarks:
        if len(landmark) > 0:
            landmark[..., 0] = landmark[..., 0]*w
            landmark[..., 1] = landmark[..., 1]*h
    return landmarks


def extract_faces(video_path, save_path, annotation=None, mask=None):
    print(f"The video {os.path.basename(video_path)} has started processing")

    batch_counter = 0
    count = 1
    side = ''
    if annotation is None:
        side = None
    else:
        # take either "right" or "left" word from annotation path
        if "_right" in os.path.basename(annotation):
            side = "right"
        if "_left" in os.path.basename(annotation):
            side = "left"

    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_w = 256
    if scaled_w < w:
        scaled_h = h * (scaled_w/w)
    else:
        scaled_h = h
    batch = []
    names = []
    initial_images = []

    for frame_idx in range(total_frames):
        if batch_counter == batch_size or frame_idx == total_frames:
            inp = torch.stack(batch).cuda().permute(0, 3, 1, 2)
            pred = fa.get_landmarks_from_batch(inp)
            if np.array(pred, dtype=np.object).size == 0:
                initial_images = []
                names = []
                batch = []
                batch_counter = 0
                continue
            # if third_person:
            #     filter_third_person(pred)
            if side is not None:
                pred = get_one_face(pred, side)
            norm_pred = norm_landmarks(pred, scaled_h, scaled_w, )
            pred = denorm_landmarks(norm_pred, h, w)

            faces = batch_wrapper(initial_images, pred)
            save_faces(faces, names, save_path)
            initial_images=[]
            names = []
            batch = []
            batch_counter = 0
        else:
            names.append(f"{count}.jpg")
            batch_counter += 1
            success, image = vidcap.read()
            if mask is not None:
                image = image*mask
                # plt.imshow(image.astype(int))
                # plt.show()

            initial_images.append(image)
            resized_image = resize(image, scaled_w)

            batch.append(torch.tensor(resized_image))
            # print('Read a new frame: ', count)
            count += 1

    print(f"The video {os.path.basename(video_path)} has ended processing")


def parse_ignores(ignore_path, video_dirs):
    ignore = {}
    masks = []

    with open(ignore_path, 'r') as file:
        data = file.readlines()
    for video in data:
        video, bboxes = video.split(";", maxsplit=1)
        bboxes = bboxes[:-1].split(";")
        for i, box in enumerate(bboxes):
            bboxes[i] = eval(box)
        ignore.update({video: bboxes})

    for video in video_dirs:
        video_name = os.path.basename(video)
        if video_name in ignore.keys():
            vid = cv2.VideoCapture(video)
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            mask = np.ones((height, width, 3))
            for box in ignore[video_name]:
                mask[box[1]:box[3], box[0]:box[2]] = 0
            masks.append(mask)
        else:
            masks.append(None)

    return masks



if __name__ == '__main__':

    # Parser

    parser = ArgumentParser()
    parser.add_argument("--video_dir", "-v", required=True, type=str,
                        help="Path to directory with videos")
    parser.add_argument("--save_dir", "-s", required=True, type=str,
                        help="Path to directory to save frames")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Batch size for network")
    parser.add_argument("--anno_dir", "-a", type=str,
                        help="Path to AffWild2 annotations if person side is necessary")
    parser.add_argument("--ignore", "-i", type=str,
                        help="Path to json file with ignore regions for videos")
    parser.add_argument("--workers", type=int)

    args = parser.parse_args()

    batch_size = args.batch_size
    video_dir = args.video_dir
    save_dir = args.save_dir
    annotation_dir = args.anno_dir
    ignore = args.ignore
    workers = args.workers

    # Directory preprocessing

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    video_dirs = os.listdir(video_dir)
    video_dirs = sorted([os.path.join(video_dir, video) for video in video_dirs if video.endswith((".mp4", ".avi"))])
    save_dirs = sorted([os.path.join(save_dir, os.path.basename(name)[:-4]) for name in video_dirs])

    if ignore is not None:
        masks = parse_ignores(ignore, video_dirs)
    else:
        masks = [None] * len(video_dirs)

    if annotation_dir is not None:
        annotation_dirs = sorted([os.path.join(annotation_dir, anno) for anno in os.listdir(annotation_dir) if anno.endswith(".txt")])
        assert len(annotation_dirs) == len(video_dirs) == len(save_dirs)
    else:
        annotation_dirs = [None] * len(video_dirs)

    # Create args for multiprocessing

    args = []

    for a, b, c, d in zip(video_dirs, save_dirs, annotation_dirs, masks):
        args.append([a, b, c, d])

    for path in save_dirs:
        if not os.path.exists(path):
            os.mkdir(path)

    if workers is not None:
        processes = workers
    else:
        processes = len(video_dirs) if len(video_dirs) < 3 else 3

    # Preparation for face aligner

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    reference_5pts = get_reference_facial_points(
        (112, 112), inner_padding_factor, outer_padding, default_square
    )

    # create face aligner object

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector="sfd")

    # start face extraction

    start_time = time.time()

    Parallel(n_jobs=processes, prefer="processes", verbose=100)\
        (delayed(extract_faces)(arg[0], arg[1], arg[2], arg[3]) for arg in args)

    print(f"{len(video_dirs)} videos are done in {(time.time() - start_time)} sec.")