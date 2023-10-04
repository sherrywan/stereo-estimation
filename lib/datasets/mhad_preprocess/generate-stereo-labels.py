'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-13 15:06:36
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-08-25 11:21:32
FilePath: /wxy/3d_pose/stereo-estimation/lib/dataset/mhad_preprocess/generate-stereo-labels.py
Description: generate label.npy of MHAD_B stereo dataset.
'''

import os
import sys
import numpy as np
import math
import h5py

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from lib.utils import rectification


def camera_parameters(camera_name, jnt=None):
    '''camera_parameters

    Args:
        camera_name (string): '1' or '2'
        jnt (numpy): shape(35, 3)

    Returns:
        R, t, K: if jnt is None
        jnt_2d: if jnt is not None
    '''
    # wrd to cluster
    R_cluster1 = np.array([0.895701051, 0.002872461, -0.444647521, 0.050160084, -0.994248986, 0.094619878, -0.441818565, -0.107054681, -0.890693903]).reshape(3,3)
    t_cluster1 = np.array([-654.69244384, 1101.511840820, 3154.582519531]).reshape(3,1)
    
    # cluster to camera
    if camera_name == '1':
        K = np.array([523.62481689, 0., 351.50247192, 0., 524.78002930, 225.29899597, 0., 0., 1.]).reshape(3,3)
        R = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
        t = np.array([0,0,0]).reshape(3,1)
        d = np.array([-0.30726913, 0.11957706, 7.77966125e-006, -1.81283604e-003]).reshape(4,1)
    elif camera_name == '2':
        K = np.array([532.78863525, 0., 320.36187744, 0., 533.49108887, 231.61149597, 0., 0., 1.]).reshape(3,3)
        R = np.array([0.9942,-0.0290,0.1034,0.0314,0.9993,-0.0222,-0.1027,0.0254,0.9944]).reshape(3,3)
        t = np.array([-113.2355,-3.1258,5.3590]).reshape(3,1)
        d = np.array([-0.30525380, 0.11849667, -2.24173709e-005, -1.13707327e-003]).reshape(4,1)
    elif camera_name == '3':
        K = np.array([535.76379395, 0., 351.78421021, 0., 536.48388672, 258.91003418, 0., 0., 1.]).reshape(3,3)
        R = np.array([0.9943,-0.0329,0.1012,0.0391,0.9974,-0.0599,-0.0990,0.0635,0.9931]).reshape(3,3)
        t = np.array([-225.5759,-8.3706,10.4491]).reshape(3,1)
        d = np.array([-0.30206695, 0.10986967, 2.87068815e-005, -5.85383852e-004]).reshape(4,1)
    elif camera_name == '4':
        K = np.array([541.68511963, 0., 334.84011841, 0., 542.19091797, 229.06407166, 0., 0., 1.]).reshape(3,3)
        R = np.array([0.9838,-0.0182,0.1781,0.0205,0.9997,-0.0110,-0.1778,0.0145,0.9840]).reshape(3,3)
        t = np.array([-331.1895,-5.8068,43.4163]).reshape(3,1)
        d = np.array([-0.30782333, 0.12185945, -6.21398794e-004, -6.72762864e-004]).reshape(4,1)

    # wrd to camera
    z = np.array([0,0,0,1]).reshape(1,4)
    T_cluster1 = np.hstack((R_cluster1, t_cluster1))
    T_cluster1 = np.vstack((T_cluster1, z))
    T = np.hstack((R, t))
    T = np.vstack((T, z))
    T_fin = T@T_cluster1
    T_fin = T_fin[:3]
    
    if jnt is None:
        return T_fin[:,:3], T_fin[:,3], K
    else:
        # wrd to img
        P = K@T_fin   
        # projection
        jnt3d_H = np.ones((jnt3d.shape[0],4))
        jnt3d_H[:,:3] = jnt3d
        jnt2d_H = (P@jnt3d_H[:,:,np.newaxis]).reshape(-1,3)
        jnt2d = jnt2d_H[:,:2] / jnt2d_H[:,2][:,np.newaxis]
        return jnt2d


def rec_the_point(point, H):
    '''rec the point

    Args:
        point (numpy):  shape (N,2)
        H

    Returns:
        point (numpy): shape (N,2)
    '''
    point_H = np.ones((*point.shape[:-1],3))
    point_H[:,:-1] = point
    point_rec_H = np.matmul(H,point_H[:,:,np.newaxis]).reshape(*point_H.shape)
    point_rec = point_rec_H[:,:-1]/(point_rec_H[:,-1][:,np.newaxis])
    return point_rec


def rec_the_bbox(bbox, H):
    '''rec the bbox

    Args:
        bbox (_type_): tlbr
        H 

    Returns:
        _type_: _description_
    '''
    p_lt = np.array([bbox[1], bbox[0]]).reshape(1,2)
    p_rb = np.array([bbox[3], bbox[2]]).reshape(1,2)

    p_lt_rec = rec_the_point(p_lt, H)[0]
    p_rb_rec = rec_the_point(p_rb, H)[0]

    top = p_lt_rec[1]
    left = p_lt_rec[0]
    bottom = p_rb_rec[1]
    right = p_rb_rec[0]

    return np.array([top,left,bottom,right])


def stereo_the_bbox(bbox):
    '''stereo the bbox

    Args:
        bbox (_type_): (bbox_right, bbox_left)

    Returns:
        _type_: _description_
    '''
    bbox2, bbox1 = bbox
    if bbox1[0]==0 and bbox1[1]==0 and bbox1[2]==0 and bbox1[3]==0:
        return bbox2
    elif bbox2[0]==0 and bbox2[1]==0 and bbox2[2]==0 and bbox2[3]==0:
        return bbox1

    top = min(bbox2[0], bbox1[0])
    left = min(bbox2[1], bbox1[1])
    bottom = max(bbox2[2], bbox1[2])
    right = max(bbox2[3], bbox1[3])

    return [top, left, bottom, right]


def square_the_bbox(bbox):
    top, left, bottom, right = bbox

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

    return top, left, bottom, right


def square_the_bbox_stereo(bbox):
    '''square the bbox stereo

    Args:
        bbox (_type_): (bbox_left, bbox_right)

    Returns:
        _type_: _description_
    '''
    bbox1, bbox2 = bbox
    
    top1, left1, bottom1, right1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    top2, left2, bottom2, right2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    if top1==0 and left1==0 and bottom1==0 and right1==0:
        return np.zeros_like(bbox)
    elif top2==0 and left2==0 and bottom2==0 and right2==0:
        return np.zeros_like(bbox)

    top = int(round(max(top2, top1)))
    bottom = int(round(min(bottom2, bottom1)))

    width = int(math.ceil(max(right2 - left2, right1 - left1)))
    height = int(math.ceil(bottom - top))

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
        center1 = (left1 + right1) * 0.5
        left1 = int(round(center1 - width * 0.5))
        right1 = left1 + width
        center2 = (left2 + right2) * 0.5
        left2 = int(round(center2 - width * 0.5))
        right2 = left2 + width
    else:
        center1 = (left1 + right1) * 0.5
        left1 = int(round(center1 - height * 0.5))
        right1 = left1 + height
        center2 = (left2 + right2) * 0.5
        left2 = int(round(center2 - height * 0.5))
        right2 = left2 + height

    return np.array([[top,left1,bottom,right1], [top,left2,bottom,right2]])


BBOXES_SOURCE = 'GT'
camera_left_index = 0
camera_right_index = 1

mhad_root = sys.argv[1]
baseline_width = sys.argv[2] # 's' or 'm' or 'l'
save_tag = sys.argv[3] # "1" - save, "0" - not

if baseline_width == 's':
    camera_list = ['1', '2']
elif baseline_width == 'm':
    camera_list = ['1', '3']
elif baseline_width == 'l':
    camera_list = ['1', '4']
destination_file_path = os.path.join(
    mhad_root, "extra", f"mhad-stereo-{baseline_width}-labels-{BBOXES_SOURCE}bboxes.npy")
destination_recmap_file_path = os.path.join(
    mhad_root, "extra", f"mhad-stereo-{baseline_width}-recmap.npy")

retval = {
    'subject_names': ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12'],
    'camera_names': camera_list, # ['1', '2'] - small baseline, ['1', '3'] - middle baseline, ['1', '4'] - large baseline
    'action_names': ['A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','A11'],
    'reptition_names': ['R01','R02','R03','R04','R05']
}
retval['cameras'] = np.empty(
    len(retval['camera_names']),
    dtype=[('R', np.float32, (3, 3)), ('t', np.float32, (3, 1)), 
           ('K', np.float32, (3, 3)), ('H', np.float32, (3, 3))])
table_dtype = np.dtype([
    ('subject_idx', np.int8),
    ('action_idx', np.int8),
    ('reptition_idx', np.int8),
    ('frame_idx', np.int16),
    ('keypoints', np.float32, (17, 3)),  # roughly MPII format
    ('keypoints_2d', np.float32, (len(retval['camera_names']), 17, 3)),  #dim3 is for visibale
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']), 4))
])
retval['table'] = [] 
recmap={}

for camera_idx, camera in enumerate(retval['camera_names']):
    camera_retval = retval['cameras'][camera_idx]
    R, t, K = camera_parameters(camera)
    camera_retval['R'] = R
    camera_retval['t'] = t[:,np.newaxis]
    camera_retval['K'] = K
 
camera_left = retval['cameras'][camera_left_index]
camera_right = retval['cameras'][camera_right_index]
camera_left['K'], camera_right['K'], camera_left['R'], camera_right[
    'R'], camera_left['t'], camera_right['t'], camera_left['H'], camera_right[
        'H'], mapx_1, mapy_1, mapx_2, mapy_2 = rectification.rectification_calculation(
            camera_left['K'], camera_right['K'],
            np.hstack([camera_left['R'], camera_left['t']]),
            np.hstack([camera_right['R'], camera_right['t']]), dims1=(640,480), dims2=(648,480))
camera_left['K'] = camera_left['K']/camera_left['K'][2,2]
camera_right['K'] = camera_right['K']/camera_right['K'][2,2]
recmap[retval['camera_names'][camera_left_index]] = [mapx_1, mapy_1]
recmap[retval['camera_names'][camera_right_index]] = [mapx_2, mapy_2]

# save recmap dict
if save_tag == "1":
    np.save(destination_recmap_file_path, recmap)

# # Fill bounding boxes
# bboxes_stereo = {}
# H_left = retval['cameras'][camera_left_index]['H']
# H_right = retval['cameras'][camera_right_index]['H']

# for subject_idx, subject in enumerate(retval['subject_names']):
#     root_folder = os.path.join(mhad_root, "Cluster01")
#     bboxes_stereo_subject = {}

#     for action_idx, action in enumerate(retval['action_names']):
#         bboxes_stereo_action = {}

#         for reptition_idx, reptition in enumerate(retval['reptition_names']):
#             if (subject_idx == 3) and (action_idx == 7) and (reptition_idx == 4) :
#                 continue
#             images_path = os.path.join(root_folder, 'Cam01', subject, action, reptition)
#             frame_idxs = sorted([int(name[-9:-4]) for name in os.listdir(images_path)])
#             bboxes_stereo_reptition = {}

#             for frame in frame_idxs:
#                 jnt_fi = os.path.join(mhad_root, "Skeleton/jnt_s%02d_a%02d_r%02d_%05d.txt"%(subject_idx+1,action_idx+1,reptition_idx,frame))
#                 jnt3d = np.loadtxt(jnt_fi)
#                 jnt2d_left = camera_parameters(retval['camera_names'][camera_left_index], jnt3d)
#                 jnt2d_right = camera_parameters(retval['camera_names'][camera_right_index], jnt3d)
                
#                 bbox_left = np.array([np.min(jnt2d_left[:,1])-20, np.min(jnt2d_left[:,0])-20,np.max(jnt2d_left[:,1])+20,np.max(jnt2d_left[:,0])+20]).reshape(1,4)
#                 bbox_right = np.array([np.min(jnt2d_right[:,1])-20, np.min(jnt2d_right[:,0])-20,np.max(jnt2d_right[:,1])+20,np.max(jnt2d_right[:,0])+20]).reshaoe(1,4)
                
#                 bbox_stereo = np.concatenate((bbox_left, bbox_right), axis=0)
#                 bbox_stereo = rec_the_bbox(bbox_stereo, H_left, H_right)
#                 bbox_stereo = square_the_bbox_stereo(bbox_stereo)
                
#                 bboxes_stereo_reptition[frame] = bbox_stereo
#             bboxes_stereo_action[reptition] = bboxes_stereo_reptition
#         bboxes_stereo_subject[action] = bboxes_stereo_action
#     bboxes_stereo[subject] = bboxes_stereo_subject

# save labels
H_left = retval['cameras'][camera_left_index]['H']
H_right = retval['cameras'][camera_right_index]['H']
for subject_idx, subject in enumerate(retval['subject_names']):
    root_folder = os.path.join(mhad_root, "Cluster01")
    for action_idx, action in enumerate(retval['action_names']):
        for reptition_idx, reptition in enumerate(retval['reptition_names']):
            if (subject_idx == 3) and (action_idx == 7) and (reptition_idx == 4) :
                continue
        
            images_path_left = os.path.join(root_folder, 'Cam%02d'%(int(retval['camera_names'][camera_left_index])), subject, action, reptition)
            images_path_right = os.path.join(root_folder, 'Cam%02d'%(int(retval['camera_names'][camera_right_index])), subject, action, reptition)
            if os.path.isdir(images_path_left) and os.path.isdir(images_path_right):
                frame_idxs_left = sorted([int(name[-9:-4]) for name in os.listdir(images_path_left)])
                frame_idxs_right = sorted([int(name[-9:-4]) for name in os.listdir(images_path_right)])
                assert frame_idxs_left == frame_idxs_right, 'frame numbers are different in %s and %s'%(images_path_left, images_path_right)
                frame_idxs = frame_idxs_left
                
                # filling keypoints3d, keypoints2d, bounding box
                jnts_3d = []
                jnts_2d_left = []
                jnts_2d_right = []
                bboxes_stereo = []
                for frame in frame_idxs:
                    jnt_fi = os.path.join(mhad_root, "Skeleton/jnt_s%02d_a%02d_r%02d_%05d.txt"%(subject_idx+1,action_idx+1,reptition_idx+1,frame))
                    jnt3d = np.loadtxt(jnt_fi)
                    
                    jnt2d_left = camera_parameters(retval['camera_names'][camera_left_index], jnt3d)
                    jnt2d_left_v = np.ones((jnt2d_left.shape[0], 1))
                    jnt2d_left_v = np.where((jnt2d_left[:,0]<0)[:,np.newaxis], 0, jnt2d_left_v)
                    jnt2d_left_v = np.where((jnt2d_left[:,0]>640)[:,np.newaxis], 0, jnt2d_left_v)
                    jnt2d_left_v = np.where((jnt2d_left[:,1]<0)[:,np.newaxis], 0, jnt2d_left_v)
                    jnt2d_left_v = np.where((jnt2d_left[:,1]>480)[:,np.newaxis], 0, jnt2d_left_v)
                    jnt2d_left_rec = rec_the_point(jnt2d_left, H_left)
                    
                    jnt2d_right = camera_parameters(retval['camera_names'][camera_right_index], jnt3d)
                    jnt2d_right_v = np.ones((jnt2d_right.shape[0], 1))
                    jnt2d_right_v = np.where((jnt2d_right[:,0]<0)[:,np.newaxis], 0, jnt2d_right_v)
                    jnt2d_right_v = np.where((jnt2d_right[:,0]>640)[:,np.newaxis], 0, jnt2d_right_v)
                    jnt2d_right_v = np.where((jnt2d_right[:,1]<0)[:,np.newaxis], 0, jnt2d_right_v)
                    jnt2d_right_v = np.where((jnt2d_right[:,1]>480)[:,np.newaxis], 0, jnt2d_right_v)
                    jnt2d_right_rec = rec_the_point(jnt2d_right, H_right)
       
                    bbox_left = np.array([np.min(jnt2d_left[:,1])-20, np.min(jnt2d_left[:,0])-20,np.max(jnt2d_left[:,1])+20,np.max(jnt2d_left[:,0])+20]).reshape(1,4)
                    bbox_right = np.array([np.min(jnt2d_right[:,1])-20, np.min(jnt2d_right[:,0])-20,np.max(jnt2d_right[:,1])+20,np.max(jnt2d_right[:,0])+20]).reshape(1,4)
                    
                    bbox_stereo = np.concatenate((bbox_left, bbox_right), axis=0)
                    bbox_stereo[0] = rec_the_bbox(bbox_stereo[0], H_left)
                    bbox_stereo[1] = rec_the_bbox(bbox_stereo[1], H_right)
                    bbox_stereo = square_the_bbox_stereo(bbox_stereo)
                    
                    jnts_3d.append(jnt3d)
                    jnts_2d_left.append(np.hstack((jnt2d_left_rec, jnt2d_left_v)))
                    jnts_2d_right.append(np.hstack((jnt2d_right_rec, jnt2d_right_v)))
                    bboxes_stereo.append(bbox_stereo)
                
                # remove frames missing bbox
                for len_idx in range(len(frame_idxs)-1, -1, -1): 
                    bbox_check = bboxes_stereo[len_idx]
                    if (bbox_check[0][2] == bbox_check[0][0]) or (bbox_check[0][1] == bbox_check[0][3]) or (bbox_check[1][2] == bbox_check[1][0]) or (bbox_check[1][1] == bbox_check[1][3]):
                        frame_idxs.pop(len_idx)
                        jnts_3d.pop(len_idx)
                        jnts_2d_left.pop(len_idx)
                        jnts_2d_right.pop(len_idx)
                        bboxes_stereo.pop(len_idx)

            # 17 joints in MPII order
            valid_joints = (25, 23, 21, 28, 30, 32, 0, 3, 4, 6, 13, 10, 8, 15, 17, 19, 5)
            poses_world = np.array(jnts_3d)[:, valid_joints]
            jnts_2d = np.concatenate((np.array(jnts_2d_left)[:, np.newaxis, :, :], np.array(jnts_2d_right)[:, np.newaxis, :, :]), axis=1)
            poses_img = jnts_2d[:,:,valid_joints]

            table_segment = np.empty(len(frame_idxs), dtype=table_dtype)
            table_segment['subject_idx'] = subject_idx
            table_segment['action_idx'] = action_idx
            table_segment['reptition_idx'] = reptition_idx
            table_segment['frame_idx'] = frame_idxs
            table_segment['keypoints'] = poses_world
            table_segment['keypoints_2d'] = poses_img
            table_segment['bbox_by_camera_tlbr'] = np.array(bboxes_stereo)

            retval['table'].append(table_segment)

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print("Total frames in MHAD stereo dataset:", len(retval['table']))
if save_tag == "1":
    np.save(destination_file_path, retval)