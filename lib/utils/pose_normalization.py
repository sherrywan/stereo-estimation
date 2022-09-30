import numpy as np
import torch
import os
# from lib.utils import vis
from lib.utils.rotation import rodriguez, rodriguez_torch
from matplotlib import pyplot as plt


# def vis_3d_pose(file_folder, keypoints_gt, keypoints_pred=None, step=10000):
#     nums = keypoints_gt.shape[0]
#     print(nums)
#     for i in range(0, nums, step):
#         fig = plt.figure(figsize=(16, 6), dpi=600)
#         axes = fig.subplots(nrows=1,
#                             ncols=4,
#                             subplot_kw=dict(
#                                 fc='whitesmoke',
#                                 projection='3d',
#                             ))
#         if keypoints_pred is not None:
#             vis.draw_3d_pose_in_two_views(axes,
#                                           keypoints_3d_gt=keypoints_gt[i],
#                                           keypoints_3d_pred=keypoints_pred[i])
#         else:
#             vis.draw_3d_pose_in_two_views(axes,
#                                           keypoints_3d_gt=keypoints_gt[i])
#         # print(keypoints_gt[i])
#         # print(keypoints_pred[i])
#         plt.savefig("%s/3d_pose_%04d.jpg" % (file_folder, i))
#         plt.close()
#     print("INFO: successfully save 3d_pose vis pictures.")


def ori_normalization(keypoints_3d_batch, vis=False, file_path=None):
    '''normalize pose orientation

    Args:
        keypoints_3d_batch (numpy): shape(batch_size, J, 3) 3:(x,z,y)
    '''
    print("keypoints_3d_batch.shape:", keypoints_3d_batch.shape)
    ori_norm = [0, 1, 0]  # 归一化朝向
    batch_size = keypoints_3d_batch.shape[0]
    for batch_i in range(batch_size):
        keypoints_3d = keypoints_3d_batch[batch_i]
        Rsho = keypoints_3d[12]
        Lsho = keypoints_3d[13]
        Hip = keypoints_3d[6]
        Lsho_H = Lsho - Hip
        Rsho_H = Rsho - Hip
        ori = np.cross(Lsho_H, Rsho_H)  # pose的朝向
        ori[2] = 0  # pose的绕着y轴的朝向
        R = rodriguez(ori, ori_norm)
        keypoints_3d_new = np.einsum('ij,mj->mi', R, keypoints_3d)
        keypoints_3d_batch[batch_i] = keypoints_3d_new
    print("keypoints_3d_batch.shape:", keypoints_3d_batch.shape)
    if file_path is not None:
        np.save(file_path, keypoints_3d_batch)

    if vis:
        print("Start drawing.")
        vis_folder = '/data0/wxy/3d_pose/holistic-triangulation/res/pose_ori'
        os.makedirs(vis_folder, exist_ok=True)
        # vis_3d_pose(vis_folder, keypoints_3d_batch, step=1000)
        print("Successfully draw orientation normalized poses.")


def ori_normalization_torch(keypoints_3d_root,
                            keypoints_3d_rshldr,
                            keypoints_3d_lshldr):
    '''normalize pose orientation (torch)

    Args:
        keypoints_3d_root (tensor): shape(3) 3:(x,z,y)
        keypoints_3d_rshldr (tensor): shape(3) 3:(x,z,y)
        keypoints_3d_lshldr (tensor): shape(3) 3:(x,z,y)

    Returns:
        R_batch (tensor): shape(batch_size, 3, 3) rotation matrix
    '''
    ori_norm = torch.tensor([0, 1, 0], device=keypoints_3d_root.device)  # 归一化朝向
    R = torch.zeros((3, 3), device=keypoints_3d_root.device)

    Rsho = keypoints_3d_rshldr
    Lsho = keypoints_3d_lshldr
    Hip = keypoints_3d_root
    Lsho_H = Lsho - Hip
    Rsho_H = Rsho - Hip
    Lsho_H = Lsho_H / torch.linalg.norm(Lsho_H)
    Rsho_H = Rsho_H / torch.linalg.norm(Rsho_H)
    ori = torch.cross(Lsho_H, Rsho_H)  # pose的朝向
    ori[2] = 0  # pose的绕着y轴的朝向

    R = rodriguez_torch(ori, ori_norm)
   
    return R


if __name__ == '__main__':
    keypoints_h36m_file_path = '../pca_data/train_data_gt/keypoints_3d_gt_re_ltorder_h36.npy'
    keypoints_3dhp_file_path = '../pca_data/train_data_gt/keypoints_3d_gt_re_ltorder_3dhp.npy'
    keypoints_h36m = np.load(keypoints_h36m_file_path)
    keypoints_3dhp = np.load(keypoints_3dhp_file_path)
    keypoints_3d = np.concatenate((keypoints_3dhp, keypoints_h36m), axis=0)
    keypoints_3d = keypoints_3d.reshape(-1, 17, 3)
    save_path = '/data0/wxy/3d_pose/holistic-triangulation/pca_data/train_data_gt/keypoints_3d_gt_re_ltorder_h363dhp_orinorm.npy'
    ori_normalization(keypoints_3d, vis=False, file_path=save_path)