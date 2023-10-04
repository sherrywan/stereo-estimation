"""
    Generate 'labels.npy' for multiview 'human36m.py'
    from https://github.sec.samsung.net/RRU8-VIOLET/multi-view-net/

    Usage: `python3 generate-labels-npy-multiview.py <path/to/Human3.6M-root> <path/to/una-dinosauria-data/h36m> <path/to/bboxes-Human36M-squared.npy>`
"""
import os
import sys
import numpy as np
import h5py
from action_to_una_dinosauria import action_to_una_dinosauria

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from lib.utils import rectification


# Change this line if you want to use Mask-RCNN or SSD bounding boxes instead of H36M's "ground truth".
BBOXES_SOURCE = 'GT'  # "GT' or 'YOLO' or 'MRCNN' or 'SSD'

camera_param_idx = [4, 2]
camera_left_index = 0
camera_right_index = 1
camera_left_name = '60457274'
camera_right_name = '55011271'
retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    # 'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'camera_names': [ '60457274', '55011271'],
    'action_names': [
        'Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2',
        'Eating-1', 'Eating-2', 'Greeting-1', 'Greeting-2', 'Phoning-1',
        'Phoning-2', 'Posing-1', 'Posing-2', 'Purchases-1', 'Purchases-2',
        'Sitting-1', 'Sitting-2', 'SittingDown-1', 'SittingDown-2',
        'Smoking-1', 'Smoking-2', 'TakingPhoto-1', 'TakingPhoto-2',
        'Waiting-1', 'Waiting-2', 'Walking-1', 'Walking-2', 'WalkingDog-1',
        'WalkingDog-2', 'WalkingTogether-1', 'WalkingTogether-2'
    ]
}
retval['cameras'] = np.empty(
    (len(retval['subject_names']), len(retval['camera_names'])),
    dtype=[('R', np.float32, (3, 3)), ('T', np.float32, (3, 1)),
           ('t', np.float32, (3, 1)), ('K', np.float32, (3, 3)),
           ('dist', np.float32, 5), ('H', np.float32, (3, 3))])

table_dtype = np.dtype([
    ('subject_idx', np.int8),
    ('action_idx', np.int8),
    ('frame_idx', np.int16),
    ('keypoints', np.float32, (17, 3)),  # roughly MPII format
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']), 4)),
    ('disparity_min', np.int16)
])
retval['table'] = []
recmap={}

h36m_root = sys.argv[1]
destination_file_path = os.path.join(
    h36m_root, "extra", f"human36m-stereo-labels-{BBOXES_SOURCE}bboxes.npy")
destination_recmap_file_path = os.path.join(
    h36m_root, "extra", f"human36m-stereo-recmap.npy")
una_dinosauria_root = sys.argv[2]
cameras_params = h5py.File(os.path.join(una_dinosauria_root, 'cameras.h5'),
                           'r')

# Fill retval['cameras']
for subject_idx, subject in enumerate(retval['subject_names']):
    for camera_idx, camera in enumerate(retval['camera_names']):
        assert len(cameras_params[subject.replace('S', 'subject')]) == 4
        camera_params = cameras_params[subject.replace(
            'S', 'subject')]['camera%d' % (camera_param_idx[camera_idx])]
        camera_retval = retval['cameras'][subject_idx][camera_idx]

        def camera_array_to_name(array):
            return ''.join(chr(int(x[0])) for x in array)

        assert camera_array_to_name(camera_params['Name']) == camera

        camera_retval['R'] = np.array(camera_params['R']).T
        camera_retval['T'] = camera_params['T']
        camera_retval['t'] = -camera_retval['R'] @ camera_params['T']

        camera_retval['K'] = 0
        camera_retval['K'][:2, 2] = camera_params['c'][:, 0]
        camera_retval['K'][0, 0] = camera_params['f'][0]
        camera_retval['K'][1, 1] = camera_params['f'][1]
        camera_retval['K'][2, 2] = 1.0

        camera_retval['dist'][:2] = camera_params['k'][:2, 0]
        camera_retval['dist'][2:4] = camera_params['p'][:, 0]
        camera_retval['dist'][4] = camera_params['k'][2, 0]

    camera_left = retval['cameras'][subject_idx][camera_left_index]
    camera_right = retval['cameras'][subject_idx][camera_right_index]
    # print("camera parameters before:")
    # print(camera_left['K'], camera_left['R'], camera_left['t'])
    # print(camera_right['K'], camera_right['R'], camera_right['t'])
    # camera_points_left = camera_left['K'][:,2]
    # camera_points_right = camera_right['K'][:,2]
    camera_left['K'], camera_right['K'], camera_left['R'], camera_right[
        'R'], camera_left['t'], camera_right['t'], camera_left['H'], camera_right[
            'H'], mapx_1, mapy_1, mapx_2, mapy_2 = rectification.rectification_calculation(
                camera_left['K'], camera_right['K'],
                np.hstack([camera_left['R'], camera_left['t']]),
                np.hstack([camera_right['R'], camera_right['t']]))
    camera_left['K'] = camera_left['K']/camera_left['K'][2,2]
    camera_right['K'] = camera_right['K']/camera_right['K'][2,2]
    # print("camera parameters after:")
    # print(camera_left['K'], camera_left['R'], camera_left['t'])
    # print(camera_right['K'], camera_right['R'], camera_right['t'])
    # print(camera_left['H'])
    # print(camera_right['H'])
    # print("camera_points_left:", camera_left['H'].dot(camera_points_left))
    # print("camera_points_right:", camera_left['H'].dot(camera_points_right))
    
    recmap[subject_idx] = ([[mapx_2, mapy_2],[mapx_1, mapy_1]])
# save recmap dict
np.save(destination_recmap_file_path, recmap)

# Fill bounding boxes
bboxes = np.load(sys.argv[3], allow_pickle=True).item()
bboxes_stereo = {}


def rec_the_bbox(bbox, H1, H2):
    '''rec the bbox

    Args:
        bbox (_type_): (bbox_left, bbox_right)
        H1 (_type_): H_left
        H2 (_type_): H_right

    Returns:
        _type_: _description_
    '''
    bbox1, bbox2 = bbox
    p_lt_1 = np.array([bbox1[1], bbox1[0], 1])
    p_rb_1 = np.array([bbox1[3], bbox1[2], 1])
    p_lt_2 = np.array([bbox2[1], bbox2[0], 1])
    p_rb_2 = np.array([bbox2[3], bbox2[2], 1])

    p_lt_rec_1 = H1.dot(p_lt_1)
    p_rb_rec_1 = H1.dot(p_rb_1)
    p_lt_rec_2 = H2.dot(p_lt_2)
    p_rb_rec_2 = H2.dot(p_rb_2)

    # top = min(p_lt_rec_1[1]/p_lt_rec_1[2], p_lt_rec_2[1]/p_lt_rec_2[2])
    # left = min(p_lt_rec_1[0]/p_lt_rec_1[2], p_lt_rec_2[0]/p_lt_rec_2[2])
    # bottom = max(p_rb_rec_1[1]/p_rb_rec_1[2], p_rb_rec_2[1]/p_rb_rec_2[2])
    # right = max(p_rb_rec_1[0]/p_lt_rec_1[2], p_rb_rec_2[0]/p_lt_rec_2[2])

    top1 = p_lt_rec_1[1]/p_lt_rec_1[2]
    left1 = p_lt_rec_1[0]/p_lt_rec_1[2]
    bottom1 = p_rb_rec_1[1]/p_rb_rec_1[2]
    right1 = p_rb_rec_1[0]/p_rb_rec_1[2]

    top2 = p_lt_rec_2[1]/p_lt_rec_2[2]
    left2 = p_lt_rec_2[0]/p_lt_rec_2[2]
    bottom2 = p_rb_rec_2[1]/p_rb_rec_2[2]
    right2 = p_rb_rec_2[0]/p_rb_rec_2[2]

    return np.array([[top1,left1,bottom1,right1],[top2,left2,bottom2,right2]])


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

    top2, left2, bottom2, right2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    top1, left1, bottom1, right1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]

    if top2==0 and left2==0 and bottom2==0 and right2==0:
        return np.zeros_like(bbox)
    elif top1==0 and left1==0 and bottom1==0 and right1==0:
        return np.zeros_like(bbox)

    top = max(top2, top1)
    bottom = min(bottom2, bottom1)

    width = max(right2 - left2, right1 - left1)
    height = bottom - top

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

    return np.array([[top,left1,bottom,right1],[top,left2,bottom,right2]])

# for subject in bboxes.keys():
#     for action in bboxes[subject].keys():
#         for camera, bbox_array in bboxes[subject][action].items():
#             for frame_idx, bbox in enumerate(bbox_array):
#                 bbox[:] = square_the_bbox(bbox)
        

for subject in bboxes.keys():
    subject_idx = 0
    for idx, subject_name in enumerate(retval['subject_names']):
        if subject_name == subject:
            subject_idx = idx
    H_left = retval['cameras'][subject_idx][camera_left_index]['H']
    H_right = retval['cameras'][subject_idx][camera_right_index]['H']
    bboxes_stereo_subject = {}
    for action in bboxes[subject].keys():
        for camera, bbox_array in bboxes[subject][action].items():
            if camera == camera_left_name:
                bboxes_left = bbox_array.reshape(-1,1,4)
            elif camera == camera_right_name:
                bboxes_right = bbox_array.reshape(-1,1,4)
        bbox_stereo = np.concatenate((bboxes_left, bboxes_right), axis=1)
        for frame_idx, bbox in enumerate(bbox_stereo):
            if subject=="S1" and action=="Phoning-2" and frame_idx==1125:
                print("")
            if BBOXES_SOURCE is 'GT':
                bbox[:] = rec_the_bbox(bbox, H_left, H_right)
            elif BBOXES_SOURCE is 'YOLO':
                bbox[:] = stereo_the_bbox(bbox)
            bbox[:] = square_the_bbox_stereo(bbox)
        bboxes_stereo_subject[action] = bbox_stereo
    bboxes_stereo[subject] = bboxes_stereo_subject


if BBOXES_SOURCE is 'MRCNN' or BBOXES_SOURCE is 'SSD' :

    def replace_gt_bboxes_with_cnn(bboxes_gt, bboxes_detected_path,
                                   detections_file_list):
        """
            Replace ground truth bounding boxes with boxes from a CNN detector.
        """
        with open(bboxes_detected_path, 'r') as f:
            import json
            bboxes_detected = json.load(f)

        with open(detections_file_list, 'r') as f:
            for bbox, filename in zip(bboxes_detected, f):
                # parse filename
                filename = filename.strip()
                filename, frame_idx = filename[:-15], int(filename[-10:-4]) - 1
                filename, camera_name = filename[:-23], filename[-8:]
                slash_idx = filename.rfind('/')
                filename, action_name = filename[:slash_idx], filename[
                    slash_idx + 1:]
                subject_name = filename[filename.rfind('/') + 1:]

                bbox, _ = bbox[:4], bbox[4]  # throw confidence away
                bbox = square_the_bbox(
                    [bbox[1], bbox[0], bbox[3] + 1,
                     bbox[2] + 1])  # LTRB to TLBR
                bboxes_gt[subject_name][action_name][camera_name][
                    frame_idx] = bbox

    detections_paths = {
        'MRCNN': {
            'train':
            "/Vol1/dbstore/datasets/Human3.6M/extra/train_human36m_MRCNN.json",
            'test':
            "/Vol1/dbstore/datasets/Human3.6M/extra/test_human36m_MRCNN.json"
        },
        'SSD': {
            'train':
            "/Vol1/dbstore/datasets/k.iskakov/share/ssd-detections-train-human36m.json",
            'test':
            "/Vol1/dbstore/datasets/k.iskakov/share/ssd-detections-human36m.json"
        }
    }

    replace_gt_bboxes_with_cnn(
        bboxes, detections_paths[BBOXES_SOURCE]['train'],
        "/Vol1/dbstore/datasets/Human3.6M/train-images-list.txt")

    replace_gt_bboxes_with_cnn(
        bboxes, detections_paths[BBOXES_SOURCE]['test'],
        "/Vol1/dbstore/datasets/Human3.6M/test-images-list.txt")

# fill retval['table']
for subject_idx, subject in enumerate(retval['subject_names']):
    subject_path = os.path.join(h36m_root, "processed", subject)
    actions = os.listdir(subject_path)
    try:
        actions.remove('MySegmentsMat')  # folder with bbox *.mat files
    except ValueError:
        pass

    for action_idx, action in enumerate(retval['action_names']):
        action_path = os.path.join(subject_path, action, 'imageSequence')
        if not os.path.isdir(action_path):
            raise FileNotFoundError(action_path)

        for camera_idx, camera in enumerate(retval['camera_names']):
            camera_path = os.path.join(action_path, camera)
            if os.path.isdir(camera_path):
                frame_idxs = sorted(
                    [int(name[4:-4]) - 1 for name in os.listdir(camera_path)])
                assert len(
                    frame_idxs
                ) > 15, 'Too few frames in %s' % camera_path  # otherwise WTF
                for len_idx in range(len(frame_idxs)-1, -1, -1): # remove missing view 
                    bbox_check = bboxes_stereo[subject][action][frame_idxs[len_idx]][0]
                    if (bbox_check[2] == bbox_check[0]) or (bbox_check[1] == bbox_check[3]):
                        frame_idxs.pop(len_idx)
                break
            
        else:
            raise FileNotFoundError(action_path)

        # 16 joints in MPII order + "Neck/Nose"
        valid_joints = (3, 2, 1, 6, 7, 8, 0, 12, 13, 15, 27, 26, 25, 17, 18,
                        19) + (14, )
        with h5py.File(
                os.path.join(
                    una_dinosauria_root, subject, 'MyPoses', '3D_positions',
                    '%s.h5' % action_to_una_dinosauria[subject].get(
                        action, action.replace('-', ' '))), 'r') as poses_file:
            poses_world = np.array(poses_file['3D_positions']).T.reshape(
                -1, 32, 3)[frame_idxs][:, valid_joints]

        table_segment = np.empty(len(frame_idxs), dtype=table_dtype)
        table_segment['subject_idx'] = subject_idx
        table_segment['action_idx'] = action_idx
        table_segment['frame_idx'] = frame_idxs
        table_segment['keypoints'] = poses_world
        table_segment[
            'bbox_by_camera_tlbr'] = 0  # let a (0,0,0,0) bbox mean that this view is missing
        table_segment['disparity_min'] = 0

        for (camera_idx, camera) in enumerate(retval['camera_names']):
            camera_path = os.path.join(action_path, camera)
            if not os.path.isdir(camera_path):
                print('Warning: camera %s isn\'t present in %s/%s' %
                      (camera, subject, action))
                continue

            for bbox, frame_idx in zip(table_segment['bbox_by_camera_tlbr'],
                                       frame_idxs):
                bbox[camera_idx] = bboxes_stereo[subject][action][frame_idx][camera_idx]
        for dis, frame_idx in zip(table_segment['disparity_min'], frame_idxs):
            dis = bboxes_stereo[subject][action][frame_idx][0] - bboxes_stereo[subject][action][frame_idx][1]


        retval['table'].append(table_segment)

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print("Total frames in Human3.6Million:", len(retval['table']))
np.save(destination_file_path, retval)
