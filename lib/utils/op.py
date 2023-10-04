import numpy as np

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.img import to_numpy, to_torch
from lib.utils import multiview, transform
import cv2


def integrate_tensor_2d(heatmaps, softmax=True):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"

    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps

    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps

    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))
    if softmax:
        heatmaps = nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(
        mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(
        mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates, heatmaps


def integrate_tensor_3d(volumes, softmax=True):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape

    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))

    mass_x = volumes.sum(dim=3).sum(dim=3)
    mass_y = volumes.sum(dim=2).sum(dim=3)
    mass_z = volumes.sum(dim=2).sum(dim=2)

    mass_times_coord_x = mass_x * torch.arange(x_size).type(torch.float).to(
        mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(y_size).type(torch.float).to(
        mass_y.device)
    mass_times_coord_z = mass_z * torch.arange(z_size).type(torch.float).to(
        mass_z.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)
    z = mass_times_coord_z.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)
        z = z / mass_z.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y, z), dim=2)
    coordinates = coordinates.reshape((batch_size, n_volumes, 3))

    return coordinates, volumes


def integrate_tensor_3d_with_coordinates(volumes,
                                         coord_volumes,
                                         softmax=True,
                                         argmax=False,
                                         summax=False):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape

    volumes = volumes.reshape((batch_size, n_volumes, -1))
    res = torch.zeros_like(volumes)
    # print("max:", torch.max(volumes))
    if argmax:
        index = torch.argmax(volumes, dim=2)
        for batch_i in range(batch_size):
            for n_vol in range(n_volumes):
                # print("argmax_before:", volumes[batch_i, n_vol, index[batch_i, n_vol]])
                # volumes[batch_i, n_vol, index[batch_i, n_vol]] *= 100
                # print("argmax_after:", volumes[batch_i, n_vol, index[batch_i, n_vol]])
                res[batch_i, n_vol, index[batch_i, n_vol]] = 1
        volumes = res
    elif softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    elif summax:
        volumes = volumes / (torch.sum(volumes, dim=2, keepdim=True) + 1e-9)
    else:
        volumes = volumes

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))
    coordinates = torch.einsum("bnxyz, xyzc -> bnc", volumes, coord_volumes)

    return coordinates, volumes


def unproject_heatmaps(heatmaps,
                       proj_matricies,
                       coord_volumes,
                       volume_aggregation_method='sum',
                       vol_confidences=None):
    device = heatmaps.device
    batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[
        0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])
    volume_shape = coord_volumes.shape[1:4]

    volume_batch = torch.zeros(batch_size,
                               n_joints,
                               *volume_shape,
                               device=device)

    # TODO: speed up this this loop
    for batch_i in range(batch_size):
        coord_volume = coord_volumes[batch_i]
        grid_coord = coord_volume.reshape((-1, 3))

        volume_batch_to_aggregate = torch.zeros(n_views,
                                                n_joints,
                                                *volume_shape,
                                                device=device)

        for view_i in range(n_views):
            # for view_i in range(1):
            heatmap = heatmaps[batch_i, view_i]
            heatmap = heatmap.unsqueeze(0)

            grid_coord_proj = multiview.project_3d_points_to_image_plane_without_distortion(
                proj_matricies[batch_i, view_i],
                grid_coord,
                convert_back_to_euclidean=False)

            invalid_mask = grid_coord_proj[:,
                                           2] <= 0.0  # depth must be larger than 0.0

            grid_coord_proj[grid_coord_proj[:, 2] == 0.0,
                            2] = 1.0  # not to divide by zero
            grid_coord_proj = multiview.homogeneous_to_euclidean(
                grid_coord_proj)

            # transform to [-1.0, 1.0] range
            grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
            grid_coord_proj_transformed[:, 0] = 2 * (
                grid_coord_proj[:, 0] / heatmap_shape[0] - 0.5)
            grid_coord_proj_transformed[:, 1] = 2 * (
                grid_coord_proj[:, 1] / heatmap_shape[1] - 0.5)
            grid_coord_proj = grid_coord_proj_transformed

            # prepare to F.grid_sample
            grid_coord_proj = grid_coord_proj.unsqueeze(1).unsqueeze(0)
            try:
                # print(heatmap, grid_coord_proj)
                current_volume = F.grid_sample(heatmap,
                                               grid_coord_proj,
                                               align_corners=True)
            except TypeError:  # old PyTorch
                current_volume = F.grid_sample(heatmap, grid_coord_proj)

            # zero out non-valid points
            current_volume = current_volume.view(n_joints, -1)
            current_volume[:, invalid_mask] = 0.0

            # reshape back to volume
            current_volume = current_volume.view(n_joints, *volume_shape)

            # collect
            volume_batch_to_aggregate[view_i] = current_volume

        # agregate resulting volume
        if volume_aggregation_method.startswith('conf'):
            volume_batch[batch_i] = (volume_batch_to_aggregate *
                                     vol_confidences[batch_i].view(
                                         n_views, n_joints, 1, 1, 1)).sum(0)
        elif volume_aggregation_method == 'sum':
            volume_batch[batch_i] = volume_batch_to_aggregate.sum(0)
        elif volume_aggregation_method == 'max':
            volume_batch[batch_i] = volume_batch_to_aggregate.max(0)[0]
        elif volume_aggregation_method == 'softmax':
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate.clone(
            )
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(
                n_views, -1)
            volume_batch_to_aggregate_softmin = nn.functional.softmax(
                volume_batch_to_aggregate_softmin, dim=0)
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(
                n_views, n_joints, *volume_shape)

            volume_batch[batch_i] = (volume_batch_to_aggregate *
                                     volume_batch_to_aggregate_softmin).sum(0)
        elif volume_aggregation_method == 'multi':
            n_views = volume_batch_to_aggregate.shape[0]
            for n_view in range(n_views):
                if n_view == 0:
                    volume_batch_softmin = volume_batch_to_aggregate[n_view]
                else:
                    volume_batch_softmin *= volume_batch_to_aggregate[n_view]
            # print("volume:", volume_batch_softmin)
            # print("volume_max:", volume_batch_softmin.max())
            # print("volume_sum:", volume_batch_softmin.sum())
            # volume_batch_softmin = volume_batch_softmin.view(n_joints, -1)
            # volume_batch_softmin = nn.functional.softmax(volume_batch_softmin, dim=-1)
            # volume_batch_softmin = volume_batch_softmin.view(n_joints, *volume_shape)
            # print("volume_max:", volume_batch_softmin.max())
            # print("volume_sum:", volume_batch_softmin.sum())
            volume_batch[batch_i] = volume_batch_softmin
        else:
            raise ValueError("Unknown volume_aggregation_method: {}".format(
                volume_aggregation_method))

    return volume_batch


def unproject_stereo_heatmap(stereo_heatmaps,
                             K_l,
                             K_r,
                             T_l,
                             T_r,
                             center_position,
                             image_shape,
                             min_dis,
                             coord_volumes,
                             vol_keypoints_3d,
                             keypoints_2d_l,
                             keypoints_2d_r):
    device = stereo_heatmaps.device
    batch_size, n_joints, heatmap_shape = stereo_heatmaps.shape[
        0], stereo_heatmaps.shape[1], tuple(stereo_heatmaps.shape[2:])
    volume_shape = coord_volumes.shape[:3]

    volume_batch = torch.zeros(batch_size,
                               n_joints,
                               *volume_shape,
                               device=device)

    for batch_i in range(batch_size):
        grid_coord = coord_volumes.reshape((-1, 3)) + center_position[batch_i]

        stereo_heatmap = stereo_heatmaps[batch_i]
        stereo_heatmap = stereo_heatmap.unsqueeze(0)

        grid_coord_proj = transform.space_trans_to_stereo(K_l[batch_i], K_r[batch_i],
                                                          T_l[batch_i], T_r[batch_i],
                                                          grid_coord,
                                                          keypoints_2d_l[batch_i],
                                                          keypoints_2d_r[batch_i])

        # tranform to stereo heatmap scale
        grid_coord_proj_transformed_1 = torch.zeros_like(grid_coord_proj)
        grid_coord_proj_transformed_1[:, 0] = grid_coord_proj[:, 0] / (image_shape[1]/ heatmap_shape[2])
        grid_coord_proj_transformed_1[:, 1] = grid_coord_proj[:, 1] / (image_shape[0]/ heatmap_shape[1])
        grid_coord_proj_transformed_1[:, 2] = grid_coord_proj[:, 2] / (image_shape[1]/ heatmap_shape[2]) - (min_dis)
        grid_coord_proj = grid_coord_proj_transformed_1

        # transform to [-1.0, 1.0] range
        grid_coord_proj_ununi = grid_coord_proj.clone()
        grid_coord_proj_transformed_2 = torch.zeros_like(grid_coord_proj)
        grid_coord_proj_transformed_2[:, 0] = 2 * (
            grid_coord_proj[:, 0] / (heatmap_shape[2]-1) - 0.5)
        grid_coord_proj_transformed_2[:, 1] = 2 * (
            grid_coord_proj[:, 1] / (heatmap_shape[1]-1) - 0.5)
        grid_coord_proj_transformed_2[:, 2] = 2 * (
            grid_coord_proj[:, 2] / (heatmap_shape[0]-1) - 0.5)
        grid_coord_proj = grid_coord_proj_transformed_2
        grid_coord_proj = grid_coord_proj.reshape(*volume_shape,3).unsqueeze(0)
        try:
            # print(heatmap, grid_coord_proj)
            current_volume = F.grid_sample(stereo_heatmap,
                                           grid_coord_proj,
                                           mode='nearest',
                                           padding_mode = 'border',
                                           align_corners=True)
        except TypeError:  # old PyTorch
            current_volume = F.grid_sample(stereo_heatmap, grid_coord_proj)

        # reshape back to volume
        current_volume = current_volume.view(n_joints, *volume_shape)

        volume_batch[batch_i] = current_volume

    return volume_batch


def gaussian_3d_relative_heatmap(keypoints_3d,
                             center_position,
                             coord_volumes,
                             std,
                             temparature=1):
    '''generate the 3d heatmap conditioned by the coord volume center position according to keypoints_3d

    keypoints_3d: 3d keypoints with shape(N,K,3)
    center_position: 3d volume center localization with shape (N,3)
    coord_volumes: 3d coordinate volumes with size from config and shape (64,64,64)
    std: 3d distance std of gaussian 

    '''
    device = keypoints_3d.device
    batch_size, n_joints = keypoints_3d.shape[0], keypoints_3d.shape[1]
    volume_shape = coord_volumes.shape[:3]

    heatmap_batch = torch.zeros(batch_size,
                               n_joints,
                               *volume_shape,
                               device=device)

    for batch_i in range(batch_size):
        grid_coord = coord_volumes.reshape((-1, 3)) + center_position[batch_i]
        
        for joint_i in range(n_joints):
            joint_3d = keypoints_3d[batch_i, joint_i].reshape(1,3)

            grid_coord_joint_dis = torch.sqrt(torch.sum((grid_coord - joint_3d)**2, dim=1))
            heatmap_i = torch.exp(-grid_coord_joint_dis**2 / (2 * std**2))
            heatmap_batch[batch_i, joint_i] = (heatmap_i * temparature).reshape(*volume_shape)

    return heatmap_batch


def gaussian_2d_pdf(coords, means, sigmas, normalize=True):
    normalization = 1.0
    if normalize:
        normalization = (2 * np.pi * sigmas[:, 0] * sigmas[:, 0])

    exp = torch.exp(-((coords[:, 0] - means[:, 0])**2 / sigmas[:, 0]**2 +
                      (coords[:, 1] - means[:, 1])**2 / sigmas[:, 1]**2) / 2)
    return exp / normalization


def render_points_as_2d_gaussians(points, sigmas, image_shape, normalize=True):
    device = points.device
    n_points = points.shape[0]

    yy, xx = torch.meshgrid(
        torch.arange(image_shape[0]).to(device),
        torch.arange(image_shape[1]).to(device))
    grid = torch.stack([xx, yy], dim=-1).type(torch.float32)
    grid = grid.unsqueeze(0).repeat(n_points, 1, 1, 1)  # (n_points, h, w, 2)
    grid = grid.reshape((-1, 2))

    points = points.unsqueeze(1).unsqueeze(1).repeat(1, image_shape[0],
                                                     image_shape[1], 1)
    points = points.reshape(-1, 2)

    sigmas = sigmas.unsqueeze(1).unsqueeze(1).repeat(1, image_shape[0],
                                                     image_shape[1], 1)
    sigmas = sigmas.reshape(-1, 2)

    images = gaussian_2d_pdf(grid, points, sigmas, normalize=normalize)
    images = images.reshape(n_points, *image_shape)

    return images


def symmetry_proportion(keypoints_3d):
    '''
    判断是否符合左右对称性
    input:
        keypoints_3d: tensor (N, 3)
    output:
        proportion: 左右不对称的累积
    '''
    h36m_body = [[[0, 1], [4, 5]], [[1, 2], [3, 4]], [[10, 11], [15, 14]],
                 [[11, 12], [13, 14]]]
    proportion = 0
    for part in h36m_body:
        part_right = part[0]
        part_left = part[1]
        length_right = torch.sqrt(
            torch.sum(
                abs(keypoints_3d[part_right[0]] -
                    keypoints_3d[part_right[1]])**2))
        length_left = torch.sqrt(
            torch.sum(
                abs(keypoints_3d[part_left[0]] -
                    keypoints_3d[part_left[1]])**2))
        propor = abs(1 - length_right / length_left)
        proportion += propor

    return proportion


def rodriguez(vec1, vec2):
    '''
    calculate rotate matrix from norm vec1 to norm vec2 (numpy)
    '''
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    # print("vec1:", vec1)
    # print("vec2:", vec2)
    theta = np.arccos(vec1 @ vec2)

    vec_n = np.cross(vec1, vec2)

    n_1 = vec_n[0]
    n_2 = vec_n[1]
    n_3 = vec_n[2]
    vec_nx = np.asarray([[0, -n_3, n_2], [n_3, 0, -n_1], [-n_2, n_1, 0]])
    c = np.dot(vec1, vec2)
    s = np.linalg.norm(vec_n)
    if s == 0:
        return np.eye(3)
    R = np.eye(3) + vec_nx + vec_nx.dot(vec_nx) * ((1 - c) / (s**2))

    # print("theta:", theta)
    # print("n:", vec_nx)
    # print("R:", R)
    return R


def rodriguez_torch(vec1, vec2):
    '''
    calculate rotate matrix from norm vec1 to norm vec2 (torch tensor)
    '''
    vec1 = vec1 / torch.linalg.norm(vec1)
    vec2 = vec2 / torch.linalg.norm(vec2)
    # print("vec1:", vec1)
    # print("vec2:", vec2)
    # theta = torch.arccos(vec1 @ vec2)

    vec_n = torch.cross(vec1, vec2)

    n_1 = vec_n[0]
    n_2 = vec_n[1]
    n_3 = vec_n[2]
    vec_nx = torch.Tensor([[0, -n_3, n_2], [n_3, 0, -n_1],
                           [-n_2, n_1, 0]]).to(vec1.device)
    c = torch.dot(vec1, vec2)
    s = torch.linalg.norm(vec_n)
    if s.item() == 0:
        return torch.eye(3, device=vec1.device)
    # print("s:", s)
    R = torch.eye(3, device=vec1.device) + vec_nx + vec_nx @ vec_nx * (
        (1 - c) / (s**2))

    # print("theta:", theta)
    # print("n:", vec_nx)
    # print("R:", R)
    return R


def kronecker(A, B):
    '''[kronecker]

    Args:
        A ([tensor (a*b)]): [input mat 1]
        B ([tensor (c*d)]): [input mat 2]

    Returns:
        [tensor]: [output mat]
    '''
    AB = torch.einsum("ab,cd -> acbd", A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    '''_summary_

    Args:
        center (_type_): bbox center
        scale (_type_): bbox scale
        rot (_type_): _description_
        output_size (_type_): image size
        shift (_type_, optional): _description_. Defaults to np.array([0, 0], dtype=np.float32).
        inv (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    '''
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def create_3d_ray_coords(camera, trans_inv, grid):
    multiplier = 1.0  # avoid numerical instability
    grid_3d = grid.clone()  # Tensor,   (hw, 2), val in 0-256
    # transform to original image R.T.dot(x.T) + T
    coords = affine_transform_pts(
        grid_3d.numpy(), trans_inv)  # array, size: (hw, 2), val: 0-1000

    K = camera['K']
    # fx, fy = 256.0 * K[0:1, 0], 256.0 * K[1:2, 1]
    # cx, cy = 256.0 * K[0:1, 2], 256.0 * K[1:2, 2]
    fx, fy = K[0:1, 0], K[1:2, 1]
    cx, cy = K[0:1, 2], K[1:2, 2]
    coords[:, 0] = (coords[:, 0] - cx[0]) / fx[0] * multiplier  # array
    coords[:, 1] = (coords[:, 1] - cy[0]) / fy[0] * multiplier

    # (hw, 3) 3D points in cam coord
    coords_cam = np.concatenate((coords, multiplier * np.ones(
        (coords.shape[0], 1))),
        axis=1)  # array

    coords_world = (camera['R'].T @ coords_cam.T +
                    camera['T']).T  # (hw, 3)    in world coordinate    array
    # coords_world = torch.from_numpy(coords_world).float()  # (hw, 3)
    return coords_world


def affine_transform_pts(pts, t):
    xyz = np.add(
        np.array([[1, 0], [0, 1], [0, 0]]).dot(pts.T), np.array([[0], [0],
                                                                 [1]]))
    return np.dot(t, xyz).T


def trans_3d_coords(keypoints_2d_img, camera_K, camera_R, camera_T):
    ''' transform keypoints location from 2d in image to 3d in world

    Args:
        keypoints_2d_img (tensor): (N,V,J,2) 
        camera_K (tensor): (N,V,3,3)
        camera_R (tensor): (N,V,3,3)
        camera_T (tensor): (N,V,3,1)

        N: batch_size  V: view num  J: pixel num

    Returns:
        keypoints_2d_world (tensor): (N,V,J,3)
    '''
    batch_size, n_views = keypoints_2d_img.shape[0:2]
    # print("keypoints_2d_img:", keypoints_2d_img)

    # image to camera
    cx = camera_K[:, :, 0:1, 2].reshape(batch_size, n_views, 1)
    cy = camera_K[:, :, 1:2, 2].reshape(batch_size, n_views, 1)
    fx = camera_K[:, :, 0:1, 0].reshape(batch_size, n_views, 1)
    fy = camera_K[:, :, 1:2, 1].reshape(batch_size, n_views, 1)
    # print("cx:", cx)
    # print("fx:", fx)
    keypoints_2d_cam = torch.zeros_like(keypoints_2d_img)
    keypoints_2d_cam[:, :, :, 0] = (keypoints_2d_img[:, :, :, 0] - cx) / fx
    keypoints_2d_cam[:, :, :, 1] = (keypoints_2d_img[:, :, :, 1] - cy) / fy
    camera_z = torch.ones((batch_size, n_views, keypoints_2d_cam.shape[2],
                           1)).to(keypoints_2d_img.device)
    keypoints_2d_cam = torch.cat((keypoints_2d_cam, camera_z), dim=-1)
    # print("keypoints_2d_cam:", keypoints_2d_cam)

    # camera to world
    keypoints_2d_world = torch.matmul(keypoints_2d_cam,
                                      camera_R) + camera_T.permute(0, 1, 3, 2)
    # print("keypoints_2d_world:", keypoints_2d_world)
    # print("cam_centor:", camera_T)

    return keypoints_2d_world


def refine_keypoints_by_neightbors(keypoints_3d, occlusion, json_limb):
    batch_size, n_joints = keypoints_3d.shape[0], keypoints_3d.shape[1]

    for batch_i in range(batch_size):
        keypoint_3d = keypoints_3d[batch_i]
        occ = occlusion[batch_i]
        # occ index
        occluded = np.nonzero(occ==1)
        for occ_i in occluded[0]:
            occ_keypoint = keypoint_3d[occ_i]
            new_keypoints = []
            for data in json_limb:
                idx = data['idx']
                name = data['name']
                childs = data['children']
                means = data['bones_mean']
                stds = data['bones_std']
                
                # if occ_i==idx:
                #     for child_i, child in enumerate(childs):
                #         if child not in occluded[0]:
                #             neighbor = keypoint_3d[child]
                #             new_keypoints.append(neighbor - ((neighbor - occ_keypoint) / torch.norm(neighbor - occ_keypoint) * means[child_i]))
                for child_i, child in enumerate(childs):
                    if occ_i==child:
                        neighbor = keypoint_3d[idx]
                        new_keypoints.append(neighbor - ((neighbor - occ_keypoint) / torch.norm(neighbor - occ_keypoint) * means[child_i]))
            new_i = 0
            new_occ = torch.zeros_like(occ_keypoint)
            for new_keypoint in new_keypoints:
                new_i += 1
                new_occ += new_keypoint 
            if new_i > 0:
                new_occ = new_occ / new_i
                # occ_keypoint = new_occ
                keypoint_3d[occ_i] = new_occ

    return keypoints_3d