import numpy as np
import cv2
import matplotlib.pyplot as plt



DEBUG = False
EYES_COORS = [36, 39, 42, 45]


def get_eyes(landmarks, dtype="int"):
    # initialize the list of (x, y)-coordinates

    ## DEBUG START:
    # if DEBUG:
    #     for person in landmarks:
    #         plt.scatter(person[:, 0], person[:, 1], 2)
    #         for i, txt in enumerate(np.linspace(0, 66, 67, dtype=int)):
    #             plt.annotate(str(txt), (person[i, 0], person[i, 1]), fontsize=12)
    #     plt.show()
    ## DEBUG END:
    landmarks = np.asarray(landmarks)[:, EYES_COORS, :]
    faces_num = landmarks.shape[0]
    eyes_coords = np.zeros_like(landmarks, dtype=dtype)

    # loop over the first 4 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for face in range(faces_num):
        for i in range(0, 4):
            eyes_coords[face, i] = (landmarks[face, i, 0], landmarks[face, i, 1])
    # return the list of (x, y)-coordinates

    return eyes_coords


def show_centers(img, left, right):

    plt.imshow(img)
    for idx in range(left.shape[0]):
        points_x = [left[idx, 0], right[idx, 0]]
        points_y = [left[idx, 1], right[idx, 1]]
        plt.scatter(points_x, points_y, 2)
    plt.show()


def get_centers(landmarks):
    EYE_LEFT_OUTTER = landmarks[:, 0]
    EYE_LEFT_INNER = landmarks[:, 1]
    EYE_RIGHT_OUTTER = landmarks[:, 2]
    EYE_RIGHT_INNER = landmarks[:, 3]

    x = landmarks[..., 0]
    y = landmarks[..., 1]
    k = []
    b = []
    for _x, _y in zip(x,y):
        A = np.vstack([_x, np.ones(len(_x))]).T
        _k, _b = np.linalg.lstsq(A, _y, rcond=None)[0]
        k.append(_k)
        b.append(_b)



    x_left = (EYE_LEFT_OUTTER[..., 0] + EYE_LEFT_INNER[..., 0]) / 2
    x_right = (EYE_RIGHT_OUTTER[..., 0] + EYE_RIGHT_INNER[..., 0]) / 2
    LEFT_EYE_CENTER = np.array([x_left, x_left * k + b], dtype=np.int32).T
    RIGHT_EYE_CENTER = np.array([x_right, x_right * k + b], dtype=np.int32).T

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER


def plot_align(imgs):
    for batch in imgs:
        for img in batch:
            plt.imshow(img)
            plt.show()


def get_aligned_face(img, left, right, side=None, w=128, h=128):
    person_num = left.shape[0]
    desired_w = w
    desired_h = h
    desired_dist = desired_w * 0.3

    eyescenter = np.array([(left[..., 0] + right[..., 0]) * 0.5, (left[..., 1] + right[..., 1]) * 0.5]).T
    dx = right[..., 0] - left[..., 0]
    dy = right[..., 1] - left[..., 1]
    dist = np.sqrt(dx * dx + dy * dy)
    # TODO: dist can be a zero
    scale = desired_dist / dist
    angle = np.degrees(np.arctan2(dy, dx))
    M = []
    for i in range(person_num):
        _M = cv2.getRotationMatrix2D(tuple(eyescenter[i]), angle[i], scale[i])
        M.append(_M)

    M = np.asarray(M)

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.4
    M[..., 0, 2] += (tX - eyescenter[..., 0])
    M[..., 1, 2] += (tY - eyescenter[..., 1])

    aligned_faces = []
    for i in range(person_num):
        aligned_face = cv2.warpAffine(img, M[i], (desired_w, desired_h))
        aligned_faces.append(aligned_face)

    return aligned_faces





