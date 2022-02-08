# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/9/21 5:30 PM
"""
import copy
import numpy as np
import math

import cv2
import transforms3d as t3d


def rot2d(angle):
    return np.asarray([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])


def affine2d():
    return np.array([[1.0, -0.1], [-0.1, 1.0]])


def rt2d(angle, x=0, y=0):
    rot = rot2d(angle)
    translation = np.asarray([[x], [y]])
    return np.concatenate([rot, translation], axis=1)


def affinert2d(angle, x=0, y=0):
    rot = rot2d(angle)
    translation = np.asarray([[x], [y]])
    affine = affine2d()
    affiner = np.matmul(rot, affine)
    affinert = np.concatenate([affiner, translation], axis=1)
    return affinert


def tsfm(img, start_points, end_points, angle=math.pi / 120, flag_vis=False):
    img_r, start_points_r, end_points_r = copy.deepcopy((img, start_points, end_points))
    start_points_r, end_points_r = np.concatenate([start_points_r, np.ones((len(start_points_r), 1))], axis=1), \
                                   np.concatenate([end_points_r, np.ones((len(end_points_r), 1))], axis=1)
    affine = affinert2d(angle, x=50, y=20)
    img_r = cv2.warpAffine(img_r, M=affine, dsize=(img_r.shape[1], img_r.shape[0]))
    start_points_r = np.dot(affine, start_points_r.T).T
    end_points_r = np.dot(affine, end_points_r.T).T
    if flag_vis:
        for start_point_r in start_points_r:
            if np.alltrue(start_point_r > 0) and np.alltrue(start_point_r < np.asarray(img_r.shape[:2])[::-1]):
                start_point_r = start_point_r.astype(int)
                print(start_point_r)
                cv2.circle(img_r, start_point_r, 10, color=(0, 0, 255), thickness=3, )
                cv2.imshow('hairs rotated', img_r)
                cv2.waitKey(0)

    return img_r, start_points_r, end_points_r


def angle_between_2_vector(v1, v2):
    assert len(v1) == len(v2)
    num_point = len(v1) if len(v1.shape) > 1 else 1
    rotation_matrix_90_counter_clockwise = np.array([[0, -1],
                                                     [1, 0]])
    v1, v2 = v1.astype(float), v2.astype(float)
    # v1, v2 = v1.reshape((-1, 2)).T, v2.reshape((-1, 2)).T
    v1, v2 = v1 / np.linalg.norm(v1, axis=-1), v2 / np.linalg.norm(v2, axis=-1)
    v1, v2 = v1.reshape((num_point, 2, 1)), v2.reshape((num_point, 2, 1))

    v1_normal = np.matmul(rotation_matrix_90_counter_clockwise, v1)
    V1 = np.concatenate([v1, v1_normal], axis=-1)

    v2_normal = np.matmul(rotation_matrix_90_counter_clockwise, v2)
    V2 = np.concatenate([v2, v2_normal], axis=-1)

    # V1, V2 = V1.reshape((-1, 2, 2)), V2.reshape((-1, 2, 2))
    rot_from_v1_2_v2 = np.matmul(V2, np.linalg.inv(V1))
    rot = np.asarray([np.eye(3) for _ in range(num_point)])
    rot[:, :2, :2] = rot_from_v1_2_v2
    angles = [t3d.euler.mat2euler(rot[i])[-1] for i in range(num_point)] if num_point > 1 else \
    t3d.euler.mat2euler(rot[0])[-1]
    # angles = rot[:, 0, 0] if num_point > 1 else rot[0, 0, 0]
    return angles


def xyxy_2_corners_coord(x_min, y_min, x_max, y_max):
    # top_left, top_right, low_right, low_left,
    img_box_coord_tgt = np.array([
        [x_min, y_min, 1],
        [x_max, y_min, 1],
        [x_max, y_max, 1],
        [x_min, y_max, 1],
    ])
    return img_box_coord_tgt


def corners_2_bounding_box_xyxy(img_box_coord_tgt):
    # assert img_box_coord_tgt[0, 0] == img_box_coord_tgt[-1, 0], 'x_min ' + str(img_box_coord_tgt[0, 0]) + ' not equals to ' + str(img_box_coord_tgt[-1, 0])
    # assert img_box_coord_tgt[0, 1] == img_box_coord_tgt[1, 1], 'y_min ' + str(img_box_coord_tgt[0, 1]) + ' not equals to ' + str(img_box_coord_tgt[1, 1])
    # assert img_box_coord_tgt[1, 0] == img_box_coord_tgt[2, 0], 'x_max ' + str(img_box_coord_tgt[1, 0]) + ' not equals to ' + str(img_box_coord_tgt[2, 0])
    # assert img_box_coord_tgt[2, 1] == img_box_coord_tgt[3, 1], 'y_max ' +  str(img_box_coord_tgt[2, 1]) + ' not equals to ' + str(img_box_coord_tgt[3, 1])

    x_min, x_max = np.min(img_box_coord_tgt[:, 0]), np.max(img_box_coord_tgt[:, 0])
    y_min, y_max = np.min(img_box_coord_tgt[:, 1]), np.max(img_box_coord_tgt[:, 1])
    return int(x_min), int(y_min), int(x_max), int(y_max)


def merge_box(boxes):
    box_biggest = boxes[0]
    for box in boxes:
        box_biggest = (min(box_biggest[0], box[0]), min(box_biggest[1], box[1]), max(box_biggest[2], box[2]), max(box_biggest[3], box[3]))
    return box_biggest


def main():
    v1, v2 = np.array([1.0,
                       0.0]), \
             np.array([1.0,
                       -0.0])
    print(angle_between_2_vector(v1, v2))


# def main():
#     xyxy = (0, 0, 100, 100)
#     corners = xyxy_2_corners_coord(*xyxy)
#     print(corners)
#     xyxy = corners_2_bounding_box_xyxy(corners)
#     print(xyxy)


if __name__ == '__main__':
    main()
