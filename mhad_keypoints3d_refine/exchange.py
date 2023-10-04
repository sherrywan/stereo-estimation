'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-09-09 22:32:02
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-09-09 22:38:58
FilePath: /wxy/3d_pose/stereo-estimation/mhad_keypoints3d_refine/yy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import numpy as np

file_path = "/data0/wxy/3d_pose/stereo-estimation/mhad_keypoints3d_refine/output_3d_joint.json"
save_path = "/data0/wxy/3d_pose/stereo-estimation/mhad_keypoints3d_refine/keypoints_3d_ppt_initial.npy"

with open(file_path, 'r') as f:
    j_data = json.load(f)

keypoints_3d = []
for k in j_data.keys():
    keypoints_3d.append(np.array(j_data[k]['pred']))

keypoints_3d = np.array(keypoints_3d)
np.save(save_path, keypoints_3d)