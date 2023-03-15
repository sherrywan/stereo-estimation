'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-15 10:38:24
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-03-15 15:03:47
FilePath: /wxy/3d_pose/stereo-estimation/lib/dataset/mhad_preprocess/rectificate-mhad.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-15 10:38:24
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-03-15 13:37:47
FilePath: /wxy/3d_pose/stereo-estimation/lib/dataset/mhad_preprocess/rectificate-mhad.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import cv2
from tqdm import tqdm
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from lib.dataset.mhad_stereo import MHADStereoViewDataset

mhad_root = os.path.join(sys.argv[1])
labels_stereo_npy_path = sys.argv[2]
recmap_stereo_npy_path = sys.argv[3]
baseline_width = sys.argv[4]
number_of_processes = int(sys.argv[5])

dataset = MHADStereoViewDataset(
    mhad_root,
    labels_stereo_npy_path,
    train=True,                       # include all possible data
    test=True,
    image_shape=None,                 # don't resize
    retain_every_n_frames_in_test=1,  # yes actually ALL possible data
    kind="mhad",
    norm_image=False,
    crop=False,                         # don't crop
    rectificated=False,
    baseline_width=baseline_width)                      

print("Dataset length:", len(dataset))

n_subjects = len(dataset.labels['subject_names'])
n_cameras = len(dataset.labels['camera_names'])

# load recmap
print("Load recmap")
recmap = np.load(recmap_stereo_npy_path, allow_pickle=True).item()

# Now the main part: undistort and rectificate images
def rectificate_and_save(idx):
    sample = dataset[idx]
    
    shot = dataset.labels['table'][idx]
    subject_idx = shot['subject_idx']
    action_idx  = shot['action_idx']
    reptition_idx = shot['reptition_idx']
    frame_idx   = shot['frame_idx']

    subject = dataset.labels['subject_names'][subject_idx]
    action = dataset.labels['action_names'][action_idx]
    reptition = dataset.labels['reptition_names'][reptition_idx]

    for camera_idx, image in enumerate(sample['images']):
        camera_name = dataset.labels['camera_names'][camera_idx]
        output_image_folder = os.path.join(
            mhad_root, 'Cluster01', 'rectificated', baseline_width, 
            'Cam%02d'%(int(camera_name)), subject, action, reptition)
        output_image_path = os.path.join(output_image_folder, 
                                         'img_l01_c%02d_s%02d_a%02d_r%02d_%05d.jpg' % (int(camera_name), subject_idx+1, action_idx+1, reptition_idx+1, frame_idx))
        os.makedirs(output_image_folder, exist_ok=True)

        mapx, mapy = recmap[camera_name]
        image_rectified = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_image_path, image_rectified)


print(f"Rectificating images using {number_of_processes} parallel processes")
cv2.setNumThreads(1)
import multiprocessing
pool = multiprocessing.Pool(number_of_processes)
for _ in tqdm(pool.imap_unordered(rectificate_and_save, range(len(dataset)), chunksize=1), total=len(dataset)):
    pass

pool.close()
pool.join()
