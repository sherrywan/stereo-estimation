'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-08-29 12:16:57
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-08-29 12:20:42
FilePath: /wxy/3d_pose/stereo-estimation/test_results.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from lib.datasets import mhad_stereo, human36m



if __name__ == '__main__':
    keypoints_2d_pred =  np.load("/data0/wxy/3d_pose/stereo-estimation/logs/h36m_13_stereo_volume_152_StereoTriangulationNet@28.08.2023-14:04:04/keypoints_2d_pred.npy")
    keypoints_2d_gt = np.load("/data0/wxy/3d_pose/stereo-estimation/logs/h36m_13_stereo_volume_152_StereoTriangulationNet@28.08.2023-14:04:04/keypoints_2d_gt.npy")
    keypoints_3d_pred = np.load("/data0/wxy/3d_pose/stereo-estimation/logs/h36m_13_stereo_volume_152_StereoTriangulationNet@28.08.2023-14:04:04/keypoints_3d_pred.npy")
    keypoints_3d_gt = np.load("/data0/wxy/3d_pose/stereo-estimation/logs/h36m_13_stereo_volume_152_StereoTriangulationNet@28.08.2023-14:04:04/keypoints_3d_gt.npy")

    dataset = human36m.Human36MMultiViewDataset(
        h36m_root = "/data1/share/dataset/human36m_multi-view/processed/",
        labels_path = "/data1/share/dataset/human36m_multi-view/extra/human36m-stereo-labels-GTbboxes.npy",
        image_shape = (384, 384),
        test=True,
        kind="human36m",
        scale_bbox=1.0    
    )
    jdr, jdr_avg = dataset.JDR_2d(keypoints_2d_pred, keypoints_2d_gt)
    print(jdr)
    print(jdr_avg)
    result, mpjpe_abs, _ = dataset.evaluate(keypoints_3d_pred)
    print(result)