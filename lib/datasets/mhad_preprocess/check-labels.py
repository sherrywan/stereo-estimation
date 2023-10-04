'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-15 13:37:12
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-03-23 22:11:13
FilePath: /wxy/3d_pose/stereo-estimation/lib/dataset/mhad_preprocess/check-labels.py
Description: check keypoints_2d, bbox in label.npy
'''

import numpy as np
import cv2
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from lib.datasets.mhad_stereo import MHADStereoViewDataset
from lib.utils import camera

mhad_root = os.path.join(sys.argv[1])
labels_stereo_npy_path = sys.argv[2]
baseline_width = sys.argv[3]

dataset = MHADStereoViewDataset(
    mhad_root,
    labels_stereo_npy_path,
    train=True,                       # include all possible data
    test=True,
    image_shape=None,                 # don't resize
    retain_every_n_frames_in_test=1,  # yes actually ALL possible data
    kind="mhad",
    scale_bbox=1.0,
    norm_image=False,
    crop=True,        
    rectificated=True,
    baseline_width=baseline_width)                      

for idx in range(0, len(dataset),5000) :
    sample = dataset[idx]
    
    shot = dataset.labels['table'][idx]
    group_idx = shot['group_idx']
    subject_idx = shot['subject_idx']
    action_idx  = shot['action_idx']
    reptition_idx = shot['reptition_idx']
    frame_idx   = shot['frame_idx']

    subject = dataset.labels['subject_names'][subject_idx]
    action = dataset.labels['action_names'][action_idx]
    reptition = dataset.labels['reptition_names'][reptition_idx]

    images = sample['images']
    keypoints_3d = sample['keypoints_3d']
    keypoints_2d = sample['keypoints_2d']
    cameras = sample['cameras']

    for camera_idx, image in enumerate(sample['images']):
        camera_name = dataset.labels['camera_names'][group_idx][camera_idx]
        image = images[camera_idx]
        image2 = image.copy()
        keypoint_2d = keypoints_2d[camera_idx]
        camera_projection = cameras[camera_idx].projection
        keypoints_3d[:,3] = 1
        keypoint_proj = np.matmul(camera_projection, keypoints_3d[:,:,np.newaxis]).reshape(-1, 3)
        keypoint_proj[:, 0] = keypoint_proj[:,0] / keypoint_proj[:,2]
        keypoint_proj[:, 1] = keypoint_proj[:,1] / keypoint_proj[:,2]
        for pt_idx, pt in enumerate(keypoint_2d):
            cv2.circle(image, (int(pt[0]), int(pt[1])), 4, (0,0,250), -1)
            cv2.putText(image, str(pt_idx), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),2)
        os.makedirs('./test', exist_ok=True)
        os.makedirs('./test/%s'%(group_idx), exist_ok=True)
        image_save_path = "./test/%s/img_l01_c%02d_s%02d_a%02d_r%02d_%05d.jpg"%(str(group_idx), int(camera_name), subject_idx+1, action_idx+1, reptition_idx+1, frame_idx)
        cv2.imwrite(image_save_path, image)
        
        for pt_idx, pt in enumerate(keypoint_proj):
            cv2.circle(image2, (int(pt[0]), int(pt[1])), 4, (0,0,250), -1)
            cv2.putText(image2, str(pt_idx), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),2)
        image_save_path = "./test/%s/img_l01_c%02d_s%02d_a%02d_r%02d_%05d_proj.jpg"%(str(group_idx), int(camera_name), subject_idx+1, action_idx+1, reptition_idx+1, frame_idx)
        cv2.imwrite(image_save_path, image2)