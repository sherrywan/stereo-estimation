import os, sys
import numpy as np
from action_to_una_dinosauria import action_to_una_dinosauria
from spacepy import pycdf
import scipy.io as scio
import cv2
import matplotlib.pyplot as plt
import pickle

def plot_heatmap(data, title, save_path=None):
    fig, ax = plt.subplots()
    c = ax.imshow(data, cmap='RdBu')
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    if save_path is not None:
        fig.savefig(save_path)
    fig.clear()
    
retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'action_names': [
        'Directions-1', 'Directions-2',
        'Discussion-1', 'Discussion-2',
        'Eating-1', 'Eating-2',
        'Greeting-1', 'Greeting-2',
        'Phoning-1', 'Phoning-2',
        'Posing-1', 'Posing-2',
        'Purchases-1', 'Purchases-2',
        'Sitting-1', 'Sitting-2',
        'SittingDown-1', 'SittingDown-2',
        'Smoking-1', 'Smoking-2',
        'TakingPhoto-1', 'TakingPhoto-2',
        'Waiting-1', 'Waiting-2',
        'Walking-1', 'Walking-2',
        'WalkingDog-1', 'WalkingDog-2',
        'WalkingTogether-1', 'WalkingTogether-2']
}
data_dir = "/data0/wxy/data/h36m/extracted/"
with open('/data0/wxy/3d_pose/H36M-Toolbox/H36M-Toolbox/camera_data.pkl', 'rb') as f:
    camera_data = pickle.load(f)
for subject_idx, subject in enumerate(retval['subject_names']):
    for action_idx, action in enumerate(retval['action_names']):
        with pycdf.CDF(os.path.join(data_dir, subject, 'Poses_D3_Positions', action_to_una_dinosauria[subject].get(action, action.replace('-', ' ')) + '.cdf')) as cdf:
            poses_3d = np.array(cdf['Pose'])
            poses_3d = poses_3d.reshape(poses_3d.shape[1], 32, 3)
        with pycdf.CDF(os.path.join(data_dir, subject, 'TOF', action_to_una_dinosauria[subject].get(action, action.replace('-', ' ')) + '.cdf')) as cdf:
            intensity = np.array(cdf[2][0])
            plot_heatmap(intensity[:,:,2], "intensity", './intensity.png')
            range = np.array(cdf[3][0])
            plot_heatmap(range[:,:,2], "range", './range.png')
            
        # for camera_idx, camera in enumerate(retval['camera_names']):
        #     segments = scio.loadmat(os.path.join(data_dir, subject, 'Segment', action_to_una_dinosauria[subject].get(action, action.replace('-', ' ')) + "."+ camera + '.mat'))
            