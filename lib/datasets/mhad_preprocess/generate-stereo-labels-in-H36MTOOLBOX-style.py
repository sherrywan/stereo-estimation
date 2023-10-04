'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-13 15:06:36
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-08-28 14:20:32
FilePath: /wxy/3d_pose/stereo-estimation/lib/dataset/mhad_preprocess/generate-stereo-labels.py
Description: generate label.npy of MHAD_B stereo dataset.
'''

import os
import sys
import numpy as np
import pickle


def camera_parameters(camera_name, jnt=None):
    '''camera_parameters

    Args:
        camera_name (string): '1' or '2'
        jnt (numpy): shape(35, 3)

    Returns:
        R, t, K, d: if jnt is None
        jnt_2d, jnt_3d_camera: if jnt is not None
    '''
    # wrd to cluster
    R_cluster1 = np.array([0.895701051, 0.002872461, -0.444647521, 0.050160084, -0.994248986, 0.094619878, -0.441818565, -0.107054681, -0.890693903]).reshape(3,3)
    t_cluster1 = np.array([-654.69244384, 1101.511840820, 3154.582519531]).reshape(3,1)
    
    # cluster to camera
    if camera_name == '1':
        K = np.array([523.62481689, 0., 351.50247192, 0., 524.78002930, 225.29899597, 0., 0., 1.]).reshape(3,3)
        R = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
        t = np.array([0,0,0]).reshape(3,1)
        d = np.array([-0.30726913, 0.11957706, 7.77966125e-006, -1.81283604e-003, 0]).reshape(5,1)
    elif camera_name == '2':
        K = np.array([532.78863525, 0., 320.36187744, 0., 533.49108887, 231.61149597, 0., 0., 1.]).reshape(3,3)
        R = np.array([0.9942,-0.0290,0.1034,0.0314,0.9993,-0.0222,-0.1027,0.0254,0.9944]).reshape(3,3)
        t = np.array([-113.2355,-3.1258,5.3590]).reshape(3,1)
        d = np.array([-0.30525380, 0.11849667, -2.24173709e-005, -1.13707327e-003, 0]).reshape(5,1)
    elif camera_name == '3':
        K = np.array([535.76379395, 0., 351.78421021, 0., 536.48388672, 258.91003418, 0., 0., 1.]).reshape(3,3)
        R = np.array([0.9943,-0.0329,0.1012,0.0391,0.9974,-0.0599,-0.0990,0.0635,0.9931]).reshape(3,3)
        t = np.array([-225.5759,-8.3706,10.4491]).reshape(3,1)
        d = np.array([-0.30206695, 0.10986967, 2.87068815e-005, -5.85383852e-004, 0]).reshape(5,1)
    elif camera_name == '4':
        K = np.array([541.68511963, 0., 334.84011841, 0., 542.19091797, 229.06407166, 0., 0., 1.]).reshape(3,3)
        R = np.array([0.9838,-0.0182,0.1781,0.0205,0.9997,-0.0110,-0.1778,0.0145,0.9840]).reshape(3,3)
        t = np.array([-331.1895,-5.8068,43.4163]).reshape(3,1)
        d = np.array([-0.30782333, 0.12185945, -6.21398794e-004, -6.72762864e-004, 0]).reshape(5,1)

    # wrd to camera
    z = np.array([0,0,0,1]).reshape(1,4)
    T_cluster1 = np.hstack((R_cluster1, t_cluster1))
    T_cluster1 = np.vstack((T_cluster1, z))
    T = np.hstack((R, t))
    T = np.vstack((T, z))
    T_fin = T@T_cluster1
    T_fin = T_fin[:3]
    
    if jnt is None:
        return T_fin[:,:3], T_fin[:,3:4], K, d
    else:
        # wrd to img
        P = K@T_fin   
        # projection
        jnt3d_H = np.ones((jnt3d.shape[0],4))
        jnt3d_H[:,:3] = jnt3d
        jnt_3d_camera = (T_fin @ jnt3d_H[:,:,np.newaxis]).reshape(-1,3)
        jnt2d_H = (P@jnt3d_H[:,:,np.newaxis]).reshape(-1,3)
        jnt2d = jnt2d_H[:,:2] / jnt2d_H[:,2][:,np.newaxis]
        return jnt2d, jnt_3d_camera


def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

def square_the_bbox(bbox):
    left, top, right, bottom = bbox

    if top==0 and left==0 and bottom==0 and right==0:
        return 0, 0, 0, 0

    width = right - left
    height = bottom - top

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height

    return np.array([left, top, right, bottom])


if __name__ == '__main__':
    mhad_root = sys.argv[1]
    baseline_width = sys.argv[2] # 's' or 'm' or 'l'
    save_tag = sys.argv[3] # "1" - save, "0" - not

    if baseline_width == 's':
        camera_list = ['1', '2']
    elif baseline_width == 'm':
        camera_list = ['1', '3']
    elif baseline_width == 'l':
        camera_list = ['1', '4']

    retval = {
        'subject_names': ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12'],
        'camera_names': camera_list, # ['1', '2'] - small baseline, ['1', '3'] - middle baseline, ['1', '4'] - large baseline
        'action_names': ['A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','A11'],
        'reptition_names': ['R01','R02','R03','R04','R05']
    }
    train_subjects = ['S01','S02','S03','S04','S05','S06','S07','S09','S10','S12']
    test_subjects = ['S08', 'S11']

    train_db = []
    test_db = []
    cnt = 0
    root_folder = os.path.join(mhad_root, "Cluster01")
    for subject_idx, subject in enumerate(retval['subject_names']):
        for action_idx, action in enumerate(retval['action_names']):
            for reptition_idx, reptition in enumerate(retval['reptition_names']):
                if (subject_idx == 3) and (action_idx == 7) and (reptition_idx == 4) :
                    continue
                for camera_idx, camera in enumerate(retval['camera_names']):
                    R, t, K, d = camera_parameters(camera)
                    camera_dict = {}
                    camera_dict['R'] = R
                    camera_dict['T'] = - R.T @ t
                    camera_dict['fx'] = K[0:1,0]
                    camera_dict['fy'] = K[1:2,1]
                    camera_dict['cx'] = K[0:1,2]
                    camera_dict['cy'] = K[1:2,2]
                    camera_dict['k'] = np.concatenate((d[0:2, 0:1], d[4:5,0:1]),axis=0)
                    camera_dict['p'] = d[2:4, 0:1]
                    
                    images_folder = os.path.join(root_folder, 'Cam%02d'%(int(retval['camera_names'][camera_idx])), subject, action, reptition)
                    if os.path.isdir(images_folder):
                        frame_idxs = sorted([int(name[-9:-4]) for name in os.listdir(images_folder)])

                    for frame in frame_idxs:
                        jnt_fi = os.path.join(mhad_root, "Skeleton/jnt_s%02d_a%02d_r%02d_%05d.txt"%(subject_idx+1,action_idx+1,reptition_idx+1,frame))
                        jnt3d = np.loadtxt(jnt_fi)
                        jnt2d, jnt3d_camera = camera_parameters(retval['camera_names'][camera_idx], jnt3d)
                        datum = {}
                        image_path = os.path.join('Cam%02d'%(int(retval['camera_names'][camera_idx])), subject, action, reptition, "img_l01_c%02d_s%02d_a%02d_r%02d_%05d.jpg"%(int(camera), subject_idx+1, action_idx+1, reptition_idx+1, frame))
                        valid_joints = [25, 23, 21, 28, 30, 32, 0, 3, 4, 6, 13, 10, 8, 15, 17, 19, 5]
                        valid_joints_h36morder = [0, 21, 23, 25, 28, 30, 32, 3, 4, 5, 6, 15, 17, 19, 8, 10, 13]
                        datum['image'] = image_path
                        datum['joints_2d'] = jnt2d[valid_joints_h36morder]
                        datum['joints_3d'] = jnt3d[valid_joints_h36morder]
                        datum['joints_3d_camera'] = jnt3d_camera[valid_joints_h36morder]
                        datum['joints_vis'] = np.ones((17, 3))
                        datum['video_id'] = cnt
                        datum['image_id'] = frame
                        datum['subject'] = subject_idx+1
                        datum['action'] = action_idx+1
                        datum['subaction'] = reptition_idx+1
                        datum['camera_id'] = int(camera)
                        datum['source'] = 'mhad'
                        datum['camera'] = camera_dict

                        jnt2d_h36m = jnt2d[valid_joints_h36morder]
                        box = np.array([np.min(jnt2d_h36m[:,0])-20, np.min(jnt2d_h36m[:,1])-20,np.max(jnt2d_h36m[:,0])+20,np.max(jnt2d_h36m[:,1])+20])

                        box = square_the_bbox(box)
                        
                        # box = _infer_box(datum['joints_3d_camera'], camera_dict, 0)
                        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                        scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                        datum['center'] = center
                        datum['scale'] = scale
                        datum['box'] = box

                        if subject in train_subjects:
                            train_db.append(datum)
                        else:
                            test_db.append(datum)

                    cnt += 1

    print("training dataset with length {}".format(len(train_db)))
    print("testing dataset with length {}".format(len(test_db)))

    if save_tag == "1":
        with open(os.path.join(mhad_root, 'extra', 'mhad_train_h36morder_bbox2dgt.pkl'), 'wb') as f:
            pickle.dump(train_db, f)

        with open(os.path.join(mhad_root, 'extra','mhad_validation_h36morder_bbox2dgt.pkl'), 'wb') as f:
            pickle.dump(test_db, f)
