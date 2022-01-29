# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/2/21 4:39 PM
"""
import cv2
import numpy as np


def draw_lines(img, info, name):
    # TODO:
    for hair_info in info:
        start, end = hair_info['follicle'], hair_info['hair_end']
        cv2.line(img, start, np.asarray(start)+1, color=(255, 0, 0), thickness=3)   # root
        cv2.line(img, end, np.asarray(start), color=(0, 255, 0), thickness=1)       # hair
        cv2.line(img, end, np.asarray(end)+1, color=(0, 0, 255), thickness=3)       # head
    cv2.imshow(name+'hairs', img)
    cv2.waitKey(0)


def draw_matches(img1, pts1, img2, pts2):
    if len(img1.shape) < 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) < 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if pts1.shape[1] != 2:
        pts1 = pts1.reshape(-1, 2)
    if pts2.shape[1] != 2:
        pts2 = pts2.reshape(-1, 2)

    # compute mapp from two image to new synthesis img
    width_synthesis = img1.shape[1] + img2.shape[1]
    height_synthesis = max(img1.shape[0], img2.shape[0])
    img_synthesis = np.zeros((height_synthesis, width_synthesis, 3))

    img_synthesis[0:img1.shape[0], 0:img1.shape[1], :] = img1
    img_synthesis[0:img2.shape[0], img1.shape[1]:, :] = img2
    # img_synthesis = img_synthesis.astype(int)
    img_synthesis = img_synthesis / 255

    pts1, pts2 = pts1.astype(int), pts2.astype(int)
    pts2[:, 0] = pts2[:, 0] + img1.shape[1]
    for i in range(len(pts1)):
        pt1, pt2 = pts1[i], pts2[i]
        color = np.random.random(3)
        # color = color.astype(int)
        img_synthesis = cv2.circle(img_synthesis, pt1, radius=15, color=color, thickness=5)
        img_synthesis = cv2.circle(img_synthesis, pt2, radius=15, color=color, thickness=5)
        img_synthesis = cv2.line(img_synthesis, pt1, pt2, color=color, thickness=2)
    return img_synthesis


def main():
    return


if __name__ == '__main__':
    main()
