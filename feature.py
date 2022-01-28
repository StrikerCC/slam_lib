# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/2/21 5:46 PM
"""
import copy
import time
from itertools import compress

import cv2
import numpy as np
from scipy.spatial.kdtree import KDTree

import read
import vis


wait_key = 0


def make_chessbaord_corners_coord(chessboard_size, square_size):
    chessbaord_corners_coord = np.zeros((chessboard_size[0] * chessboard_size[1], 3))
    xy = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    chessbaord_corners_coord[:, :2] = xy
    return chessbaord_corners_coord * square_size


def get_checkboard_corners(img, checkboard_size, flag_vis=False):
    assert len(checkboard_size) == 2
    flag_found, corners = cv2.findChessboardCorners(img, checkboard_size)
    if flag_found:
        cv2.cornerSubPix(img, corners, winSize=(5, 5), zeroZone=(-1, -1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        if flag_vis:
            cv2.namedWindow('checkboard corners', cv2.WINDOW_NORMAL)
            cv2.drawChessboardCorners(img, patternSize=checkboard_size, corners=corners, patternWasFound=True)
            cv2.imshow('checkboard corners', img)
            cv2.waitKey(0)
    return corners


def sift_features(img, flag_debug=False):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    if flag_debug:
        img_kp = cv2.drawKeypoints(img, kp, None)
        print(len(kp))
        print(kp[0])
        print(des.shape)
        cv2.namedWindow('kp', cv2.WINDOW_NORMAL)
        cv2.imshow('kp', img_kp)
        cv2.waitKey(wait_key)

    return kp, des


def match_pts(img1, img2, flag_debug=False):
    if img1 is None or img2 is None:
        return None
    kp1, des1 = sift_features(img1, flag_debug)
    kp2, des2 = sift_features(img2, flag_debug)
    bf = cv2.BFMatcher_create()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [first for first, second in matches if first.distance < 0.85 * second.distance]

    if flag_debug:
        print('Get ', len(good_matches), ' good matches')
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imshow('match', img3)
        cv2.waitKey(wait_key)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return pts1, pts2


# def match_filter_pts_pair(kp1, des1, kp2, des2):
def match_filter_pts_pair(pts1, des1, pts2, des2):
    bf = cv2.BFMatcher_create()
    matches = bf.knnMatch(des1, des2, k=2)

    '''filter with feature value ambiguity'''
    good_matches = [first for first, second in matches if first.distance < 0.85 * second.distance]
    index_match = np.asarray([[m.queryIdx, m.trainIdx] for m in good_matches])

    pts1 = pts1[index_match[:, 0].tolist()]     # queryIdx
    des1 = des1[index_match[:, 0]]

    pts2 = pts2[index_match[:, 1].tolist()]     # trainIdx
    des2 = des2[index_match[:, 1]]

    print('feature value ambiguity', len(good_matches), '/', len(matches), 'points left')

    '''filter with epi-polar geometry ransac'''
    F, mask = cv2.cv2.findFundamentalMat(pts1, pts2, cv2.cv2.FM_RANSAC, ransacReprojThreshold=2.0, confidence=0.9999,
                                         maxIters=10000)

    mask_ = mask.squeeze().astype(bool).tolist()

    index_match = index_match[mask_]
    index_match_set = set(index_match[:, 0].tolist())

    pts1 = pts1[mask_]
    des1 = des1[mask_]
    pts2 = pts2[mask_]
    des2 = des2[mask_]

    good_matches = [m for m in good_matches if m.queryIdx in index_match_set]

    print('epi-polar geometry ransac', len(good_matches), '/', len(matches), 'points left, epi-polar rms',
          np.mean(np.sum(np.matmul(np.hstack([pts1, np.ones((len(pts1), 1))]), F)*np.hstack([pts2, np.ones((len(pts1), 1))]), axis=1)))

    return pts1, des1, pts2, des2


def get_sift_and_pts(img1, img2, flag_debug=False):
    if img1 is None or img2 is None:
        return None

    shrink = 2.0
    img1_sub = cv2.resize(img1, (int(img1.shape[1] / shrink), int(img1.shape[0] / shrink)))
    img2_sub = cv2.resize(img2, (int(img2.shape[1] / shrink), int(img2.shape[0] / shrink)))

    # kp1, des1 = sift_features(img1_sub, flag_debug)
    # kp2, des2 = sift_features(img2_sub, flag_debug)

    kp1, des1 = sift_features(img1_sub)
    kp2, des2 = sift_features(img2_sub)

    pts1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 2) * shrink
    pts2 = np.float32([kp.pt for kp in kp2]).reshape(-1, 2) * shrink

    # pts1, des1, pts2, des2, good_matches = match_filter_pts_pair(kp1, des1, kp2, des2)
    pts1, des1, pts2, des2 = match_filter_pts_pair(pts1, des1, pts2, des2)

    '''statistics'''
    if flag_debug:
        print('Get ', len(pts1), ' good matches')
        img3 = vis.draw_matches(img1, pts1, img2, pts2)
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imshow('match', img3)
        cv2.waitKey(wait_key)

    return pts1, des1, pts2, des2


def get_pts_pair_by_sift(img1, img2, flag_debug=False):
    pts1, des1, pts2, des2, good_matches = get_sift_and_pts(img1, img2, flag_debug)
    return pts1, pts2


def get_pts(img):
    # TODO:
    pass


def compute_fpf(img, start_points, end_points, num_neighbor):
    """"""
    # num_neighbor = 5
    num_hair = len(start_points)

    assert start_points.shape[1] == 2 and end_points.shape[1] == 2
    '''setup local frame for each point'''
    # use hair line as first axis, counter-clockwise as positive rotation

    hair_line = end_points - start_points
    hair_line_unit_vector = hair_line / np.expand_dims(np.linalg.norm(hair_line, axis=1), axis=-1)
    assert 0.999 < np.linalg.norm(hair_line_unit_vector[0]) < 1.0001, np.linalg.norm(hair_line_unit_vector[0])

    '''build tree for hair data, key is hair start coordinate, value is hair orientation'''
    tree_start = KDTree(start_points)

    '''use orientation and distance of knn feature to make extra dimension data, start from neighbor close to first axis'''
    feature_dis_start = np.zeros((num_hair, num_neighbor))
    feature_dis_end = np.zeros((num_hair, num_neighbor))
    feature_orientation_neighbor_line = np.zeros((num_hair, num_neighbor))
    feature_orientation_neighbor_hair = np.zeros((num_hair, num_neighbor))

    for index_hair in range(num_hair):
        hair_point = start_points[index_hair]
        _, nn_index_list = tree_start.query(hair_point, k=num_neighbor+1)
        nn_index_list = nn_index_list[1:]   # get rid of point itself

        for i_nn, index_nn in enumerate(nn_index_list):
            feature_dis_start[index_hair, i_nn] = np.linalg.norm(start_points[index_hair] - start_points[index_nn])   # distance
            feature_dis_end[index_hair, i_nn] = np.linalg.norm(end_points[index_hair] - end_points[index_nn])   # distance

            feature_orientation_neighbor_line[index_hair, i_nn] = utils.angle_between_2_vector(hair_line[index_hair], hair_line[index_nn])      # orientation of line segment from hair start to neighbor hair start in local frame
            # feature_orientation_neighbor_line[index_hair, i_nn] = np.random.random(1)

        # reorganize the nn list, start from point heading close to first axis
        mask_sorted = np.argsort(feature_orientation_neighbor_line[index_hair])
        feature_dis_start[index_hair] = feature_dis_start[index_hair][mask_sorted]
        feature_dis_end[index_hair] = feature_dis_end[index_hair][mask_sorted]

        feature_orientation_neighbor_line[index_hair] = feature_orientation_neighbor_line[index_hair][mask_sorted]
        feature_orientation_neighbor_hair[index_hair] = feature_orientation_neighbor_hair[index_hair][mask_sorted]

    feature_dis_start /= np.expand_dims(np.sum(feature_dis_start, axis=1), axis=-1)                 # normalize distance
    feature_dis_end /= np.expand_dims(np.sum(feature_dis_end, axis=1), axis=-1)
    # feature_dis_start *= np.pi
    # feature_dis_end *= np.pi

    feature_orientation_neighbor_line = np.arctan(feature_orientation_neighbor_line)    # orientation of line segment from hair start to neighbor hair start in local frame
    # feature_orientation_neighbor_line /= 2 * np.pi
    # feature_orientation_neighbor_line += 0.5

    '''build new feature tree from new feature'''
    # feature = np.concatenate([feature_dis, feature_orientation_neighbor_line, feature_orientation_neighbor_hair], axis=1)
    feature = np.concatenate([feature_dis_start, feature_orientation_neighbor_line], axis=1)
    # feature = feature / np.expand_dims(np.linalg.norm(feature, axis=-1), axis=-1)
    # feature = np.concatenate([feature_dis_start, feature_dis_end], axis=1)
    return feature


def main():
    time_0 = time.time()
    vis_flag = True
    num_nn = 5
    img_l, data_l = read.format_data(img_path='/home/cheng/proj/3d/hair_host/bin/left.jpg',
                                     data_json_path='/home/cheng/proj/3d/hair_host/bin/left_hair_info.json')
    img_l_copy = copy.deepcopy(img_l)
    vis.draw_lines(img_l_copy, data_l, 'l')

    img_r, data_r = read.format_data(img_path='/home/cheng/proj/3d/hair_host/bin/right.jpg',
                                     data_json_path='/home/cheng/proj/3d/hair_host/bin/right_hair_info.json')
    img_r_copy = copy.deepcopy(img_r)
    vis.draw_lines(img_r_copy, data_r, 'r')

    start_points_l, end_points_l = read.dic_2_nparray(img_l, data_l)
    start_points_r, end_points_r = read.dic_2_nparray(img_r, data_r)
    # cv2.namedWindow('line', cv2.WINDOW_NORMAL)


    # siftfeatures = cv2.xfeatures2d.SIFT_create()
    # keypoints = siftfeatures.detect(img_l_copy, None)
    # img_l_copy = cv2.drawKeypoints(img_l_copy, keypoints=keypoints, outImage=0, color=(0, 255, 0))


    img_r, start_points_r, end_points_r = utils.tsfm(img_l, start_points_l, end_points_l)

    feature_1 = compute_fpf(img_l, start_points_l, end_points_l, num_neighbor=num_nn)
    feature_2 = compute_fpf(img_r, start_points_r, end_points_r, num_neighbor=num_nn)

    '''randomize data'''
    mask = np.arange(0, len(feature_2)).astype(int)
    # np.random.shuffle(mask)
    feature_2 = feature_2[mask]
    '''for each hair, locate it nn in new feature space'''
    num_hair = len(feature_1)
    num_success = 0
    tree_feature_2 = KDTree(feature_2)
    num_neighbor_feature = 1

    img_feature = np.concatenate([img_l, img_r], axis=1)
    for i_hair in range(num_hair):
        hair_feature = feature_1[i_hair]
        _, nn_index_list = tree_feature_2.query(hair_feature, k=num_neighbor_feature + 1)
        if mask[nn_index_list[0]] == i_hair:
            num_success += 1
        if vis_flag:
            print(i_hair)
            # for index_nn in nn_index_list:
            index_match = mask[nn_index_list[0]]
            print('     ', index_match)
            start_point_r = start_points_r[index_match].astype(int)
            start_point_r += np.asarray([img_l.shape[1], 0])
            cv2.line(img_feature, start_points_l[i_hair], start_point_r, color=(0, 0, 155), thickness=2)

    cv2.namedWindow('line', cv2.WINDOW_NORMAL)
    cv2.imshow('line', img_feature)
    cv2.waitKey(0)
        # cv2.destroyAllWindows()

    '''out'''
    print('success rate', num_success/num_hair, 'time', time.time() - time_0)


if __name__ == '__main__':
    main()
