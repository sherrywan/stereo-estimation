import cv2
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))

from lib.models.yolo import DET_YOLO, results_numpy
from lib.dataset.human36m import Human36MMultiViewDataset



h36m_root = os.path.join(sys.argv[1], "processed")
bbox_stereo_npy_folder = os.path.join(sys.argv[1], "extra")
bbox_GT_npy_path = os.path.join(sys.argv[1], 'extra/bboxes-Human36M-GT.npy')
pt_path = sys.argv[2]

det_model = DET_YOLO(model='yolov5s', verbose=True, pt_path=pt_path) # load the yolo model

retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    # 'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'camera_names': ['55011271', '60457274'],
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
bbox_results = np.load(bbox_GT_npy_path, allow_pickle=True).item()

print("start detecting")
for subject_idx, subject in enumerate(retval['subject_names']):
    subject_path = os.path.join(h36m_root, subject)
    actions = os.listdir(subject_path)
    try:
        actions.remove('MySegmentsMat')  # folder with bbox *.mat files
    except ValueError:
        pass

    for action_idx, action in enumerate(retval['action_names']):
        action_path = os.path.join(subject_path, action, 'imageSequence-rectificated-undistorted')
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
            
            bbox_frames = []
            print("frame len:", len(frame_idxs))
            for frame_idx in frame_idxs:
                image_path = os.path.join(camera_path, 'img_%06d.jpg' % (frame_idx + 1))
                assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
                image = cv2.imread(image_path)[:, :, ::-1]  # OpenCV image (BGR to RGB)
                input_img = image[:, :, ::-1]
                det_results = det_model(input_img) # detect
                bboxes = results_numpy(det_results)
                bboxes = np.rint(bboxes)
                if bboxes.size != 0:
                    bbox = bboxes[0]
                    bbox = bbox.astype(np.int_)
                else:
                    bbox = np.zeros((4))
                    print("{}-{}-{}-{} with none bboxes.".format(subject, action, camera, frame_idx))
                bbox_results[subject][action][camera][frame_idx] = np.array([bbox[1], bbox[0], bbox[3], bbox[2]])  #LTRB to TLBR


bbox_path = os.path.join(bbox_stereo_npy_folder, "bboxes-Human36M-yolo.npy")
np.save(bbox_path, bbox_results)