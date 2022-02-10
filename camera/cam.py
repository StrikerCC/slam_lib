# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/30/21 2:56 PM
"""
import json
import os
import time

import cv2
import numpy as np

import slam_lib.mapping as mapping


class PinHoleCamera:
    def __init__(self):
        self.camera_matrix = None
        self.pose = None
        self.distortion_coefficient = None
        self.map_undistort = None
        self.map_rectify = None
        self.rotation_rectify = None
        self.camera_matrix_rectify = None

    def proj(self, pts_3d):
        """
        project 3d points in world frame to pixel frame
        :param pts_3d: (n, 3)
        :type pts_3d:
        :return:
        :rtype:
        """
        if pts_3d.shape[0] != 3:
            pts_3d = pts_3d.T  # (3, N)
        assert len(pts_3d.shape) == 2 and pts_3d.shape[0] == 3

        # projection
        if self.pose:
            pts_3d_homo = np.vstack([pts_3d, np.ones(pts_3d.shape[1])])  # (4, N)
            pts_3d = np.matmul(np.linalg.inv(self.pose), pts_3d_homo)[:3, :]
        pts_2d_line = np.matmul(self.camera_matrix, pts_3d)  # (3, N)
        pts_2d = pts_2d_line[:2, :] / pts_2d_line[-1, :]  # (2, N)
        return pts_2d.T  # (2, N)

    def proj_and_distort(self, pts_3d):
        """
        project 3d points in world frame to pixel frame
        :param pts_3d: (n, 3)
        :type pts_3d:
        :return:
        :rtype:
        """
        if pts_3d.shape[0] != 3:
            pts_3d = pts_3d.T  # (3, N)
        assert len(pts_3d.shape) == 2 and pts_3d.shape[0] == 3

        pts_2d = self.proj(pts_3d)

        # distortion
        pts_2d = mapping.distort_pt_2d(self.camera_matrix, self.distortion_coefficient, pts_2d)     # (2, N)
        return pts_2d  # (2, N)

    def undistort_rectify_pts(self, pts_2d):
        """

        :param pts_2d: x, y raw coord in opencv format
        :return:
        """
        if pts_2d.shape[1] == 2:
            pts_2d = pts_2d.T

        # pts_2d = pts_2d.reshape(2, -1).astype(int)[::-1, :]
        # pts_2d_rectified = self.map_rectify[pts_2d.tolist()]

        pts_2d_rectified = cv2.undistortPoints(src=pts_2d, cameraMatrix=self.camera_matrix,
                                               distCoeffs=self.distortion_coefficient, R=self.rotation_rectify,
                                               P=self.camera_matrix_rectify)

        # pts_2d = pts_2d.reshape(2, -1).astype(float)
        #
        # pts_2d_homo = np.vstack([pts_2d, np.ones((1, pts_2d.shape[1]))])
        #
        # pts_2d_rectified = np.matmul(self.camera_matrix, pts_2d_homo)
        return pts_2d_rectified

    def undistort_rectify_img(self, img):
        return cv2.remap(img, map1=self.map_rectify[:, :, 0], map2=self.map_rectify[:, :, 1],
                         interpolation=cv2.INTER_LINEAR)


class BiCamera:
    # leftCameraMatrix, rightCameraMatrix = None, None
    # leftDistCoeffs, rightDistCoeffs = None, None
    # R, T, E, F = None, None, None, None
    # R1, P1, R2, P2, Q = None, None, None, None, None

    t = None
    r = None
    img_size = None
    map_undistort_left, map_rectify_left = None, None
    map_undistort_right, map_rectify_right = None, None

    def __init__(self, para_file_path=None):
        self.cam_left, self.cam_right = PinHoleCamera(), PinHoleCamera()
        if para_file_path is not None:
            if os.path.isfile(str(para_file_path)):
                self.read_cal_param_file_and_set_params(para_file_path)
            else:
                print('calibration file doesn\'t exist at', para_file_path)

    def read_cal_param_file_and_set_params(self, para_file_path):
        if not os.path.isfile(str(para_file_path)):
            print('calibration file read fail at', para_file_path)
            return False
        if para_file_path[-3:] == 'xml':
            # f = cv2.cv2.FileStorage(para_file_path, cv2.cv2.FILE_STORAGE_READ)
            # self.cam_left.camera_matrix, self.cam_right.camera_matrix = f.getNode('leftCameraMatrix').mat(), f.getNode('rightCameraMatrix').mat()
            # self.cam_left.distortion_coefficient, self.cam_right.distortion_coefficient = f.getNode('leftDistCoeffs').mat(), f.getNode('rightDistCoeffs').mat()
            # self.rotation, self.translation, self.essential_matrix, self.fundamental_matrix = f.getNode('R').mat(), f.getNode('T').mat(), f.getNode('E').mat(), f.getNode('F').mat()
            # self.R1, self.P1, self.R2, self.P2, self.Q = f.getNode('R1').mat(), f.getNode('P1').mat(), f.getNode('R2').mat(), f.getNode('P2').mat(), f.getNode('Q').mat()
            return False
        elif para_file_path[-4:] == 'json':
            print('read camera parameter from ', para_file_path)
            f = open(para_file_path, 'r')
            params = json.load(f)
            self.set_params(params)
            return True

    def set_params(self, params):
        print('setting camera parameters')
        assert isinstance(params, dict), type(params)
        self.img_size = params['img_size']
        self.cam_left.camera_matrix = np.asarray(params['camera_matrix_left'])
        self.cam_right.camera_matrix = np.asarray(params['camera_matrix_right'])
        self.cam_left.distortion_coefficient = np.asarray(params['distortion coefficients_left'])
        self.cam_right.distortion_coefficient = np.asarray(params['distortion coefficients_right'])
        self.r = np.asarray(params['rotation'])
        self.t = np.asarray(params['translation'])

        time_start = time.time()
        print('init undistortion and rectify mapping')
        result = cv2.stereoRectify(cameraMatrix1=self.cam_left.camera_matrix,
                                   distCoeffs1=self.cam_left.distortion_coefficient,
                                   cameraMatrix2=self.cam_right.camera_matrix,
                                   distCoeffs2=self.cam_right.distortion_coefficient, imageSize=self.img_size,
                                   R=cv2.Rodrigues(self.r)[0], T=self.t)

        rectify_rotation_left, rectify_rotation_right, rectified_camera_matrix_left, rectified_camera_matrix_right, Q, \
        roi_left, roi_right = result

        map_undistort, map_rectify = cv2.initUndistortRectifyMap(
            cameraMatrix=self.cam_left.camera_matrix,
            distCoeffs=self.cam_left.distortion_coefficient,
            R=rectify_rotation_left,
            newCameraMatrix=rectified_camera_matrix_left,
            size=self.img_size,
            m1type=cv2.CV_32FC1)
        self.cam_left.rotation_rectify = rectify_rotation_left
        self.cam_left.camera_matrix_rectify = rectified_camera_matrix_left
        self.cam_left.map_rectify = np.asarray([map_undistort, map_rectify]).transpose((1, 2, 0))

        map_undistort, map_rectify = cv2.initUndistortRectifyMap(
            cameraMatrix=self.cam_right.camera_matrix,
            distCoeffs=self.cam_right.distortion_coefficient,
            R=rectify_rotation_right,
            newCameraMatrix=rectified_camera_matrix_right,
            size=self.img_size,
            m1type=cv2.CV_32FC1)
        self.cam_right.rotation_rectify = rectify_rotation_right
        self.cam_right.camera_matrix_rectify = rectified_camera_matrix_right
        self.cam_right.map_rectify = np.asarray([map_undistort, map_rectify]).transpose((1, 2, 0))
        self.Q = Q
        print('left_camera_reprojection_error', params["left_camera_re-projection_error"])
        print("right_camera_reprojection_error", params['right_camera_re-projection_error'])
        print("stereo_camera_reprojection_error", params["stereo_camera_re-projection_error"])
        print('undistortion and rectify mapping set in ', time.time() - time_start, 'second')
        return True

    def transform_rectify_pixel_to_world_coordiante(self, pt_2d_left, pt_2d_right):
        pt_2d_left, pt_2d_right = pt_2d_left.reshape(-1, 2), pt_2d_right.reshape(-1, 2)
        disparity = pt_2d_left[:, 0] - pt_2d_right[:, 0]
        _Q = self.Q
        homg_pt = np.matmul(_Q, np.vstack([pt_2d_left[:, 0], pt_2d_left[:, 1], disparity, np.ones(len(disparity))]))
        pts = homg_pt[:-1, :]
        pts /= homg_pt[3, :]
        return pts.T

    def correspondence_to_3d_in_left_rectify(self, pts_2d_left, pts_2d_right):
        pts_2d_left, pts_2d_right = pts_2d_left.reshape(-1, 2), pts_2d_right.reshape(-1, 2)
        pts_2d_left = self.cam_left.undistort_rectify_pts(pts_2d_left)
        pts_2d_right = self.cam_right.undistort_rectify_pts(pts_2d_right)
        return self.transform_rectify_pixel_to_world_coordiante(pts_2d_left, pts_2d_right)

    def correspondence_to_3d_in_left(self, pts_2d_left, pts_2d_right):
        pts_3d_in_left_rectify = self.correspondence_to_3d_in_left_rectify(pts_2d_left, pts_2d_right)
        tf_left_2_left_rectify = mapping.rt_2_tf(self.cam_left.rotation_rectify, np.zeros((3, 1)))
        pts_3d_in_left = mapping.transform_pt_3d(np.linalg.inv(tf_left_2_left_rectify), pts_3d_in_left_rectify)
        return pts_3d_in_left


def main():
    bcmaera = BiCamera('/home/cheng/proj/3d/BiCameraSDKv2.0/test_data/saved_parameters.xml')
    pt_left, pt_right = np.array([220, 220]), np.array([100, 100])
    world = bcmaera.transform_rectify_pixel_to_world_coordiante(pt_left, pt_right)
    print(world)


if __name__ == '__main__':
    main()
