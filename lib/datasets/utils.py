'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-15 15:48:07
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-07-05 09:36:55
FilePath: /wxy/3d_pose/stereo-estimation/lib/datasets/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch

from lib.utils.img import image_batch_to_torch


def make_collate_fn_for_MaskST(randomize_n_views=True, min_n_views=10, max_n_views=31):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()

        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        batch['keypoints_2d'] = [item['keypoints_2d'] for item in items]
        # batch['cuboids'] = [item['cuboids'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]
        batch['occlusion'] = [item['occlusion'] for item in items]

        try:
            batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
        except:
            pass

        return batch

    return collate_fn


def make_collate_fn(randomize_n_views=True, min_n_views=10, max_n_views=31):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['images']) for item in items)

        indexes = np.arange(total_n_views)
        if randomize_n_views:
            n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
            indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
        else:
            indexes = np.arange(total_n_views)

        batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]
        batch['keypoints_3d_ca'] = np.array([[item['keypoints_3d_ca'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        
        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        batch['keypoints_2d'] = [item['keypoints_2d'] for item in items]
        # batch['cuboids'] = [item['cuboids'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]
        batch['occlusion'] = [item['occlusion'] for item in items]

        try:
            batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
        except:
            pass

        return batch

    return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def prepare_batch(batch, device, config, is_train=True):
    # images
    images_batch = []
    for image_batch in batch['images']:
        image_batch = image_batch_to_torch(image_batch)
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

    images_batch = torch.stack(images_batch, dim=0)

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]).float().to(device)
    keypoints_3d_batch_ca = torch.from_numpy(np.stack(batch['keypoints_3d_ca'], axis=0)[:, :, :, :3]).squeeze(dim=-1).float().to(device)
    
    # 3D keypoints validity
    keypoints_3d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, 3:]).float().to(device)

    # projection matricies
    proj_matricies_batch =  torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
    proj_matricies_batch = proj_matricies_batch.float().to(device)
    
    # camera instrinc matricies
    K_batch = torch.stack([torch.stack([torch.from_numpy(camera.getK) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 3)
    K_batch = K_batch.float().to(device)

    # camera exstrinc matricies
    R_batch = torch.stack([torch.stack([torch.from_numpy(camera.getR) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 3)
    R_batch = R_batch.float().to(device)
    t_batch = torch.stack([torch.stack([torch.from_numpy(camera.gett) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 1)
    t_batch = t_batch.float().to(device)

    # camera centors
    T_batch = torch.stack([torch.stack([torch.from_numpy(camera.getT) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 1)
    T_batch = T_batch.float().to(device)

    # 2D keypoints
    keypoints_2d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_2d'], axis=0)[:, :, :, :2]).float().to(device)
    
    # 2D keypoints validity
    keypoints_2d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_2d'], axis=0)[:, :, :, 2:]).float().to(device)

    # occlusion in left camera
    occlusion_left = torch.from_numpy(np.stack(batch['occlusion'], axis=0)[:, :, :3]).float().to(device)

    return images_batch, keypoints_3d_batch_gt, keypoints_3d_batch_ca, keypoints_3d_validity_batch_gt, keypoints_2d_batch_gt, keypoints_2d_validity_batch_gt, proj_matricies_batch, K_batch, T_batch, R_batch, t_batch, occlusion_left


def prepare_batch_for_pretrain(batch, device, config, is_train=True):

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]).float().to(device)
    
    # 3D keypoints validity
    keypoints_3d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, 3:]).float().to(device)

    return keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt