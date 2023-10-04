from copy import deepcopy
import numpy as np
import pickle
import random
import json

from scipy.optimize import least_squares

import torch
from torch import nn

from lib.utils import op, multiview, volumetric

from lib.models import pose_resnet as pose_resnet
from lib.models.v2v import V2VModel, VHModel
from lib.models.psm_gnn import PSMGNNModel, PSMGNNModel_nolearn


class RANSACTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        self.backbone = pose_resnet.get_pose_net(
            config.model.backbone, device=device)

        self.direct_optimization = config.model.direct_optimization

    def forward(self, images, proj_matricies, batch):
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integrate
        heatmaps, _, _, _ = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(
            heatmaps.shape[3:])

        # keypoints 2d
        _, max_indicies = torch.max(heatmaps.view(
            batch_size, n_views, n_joints, -1), dim=-1)
        keypoints_2d = torch.stack(
            [max_indicies % heatmap_shape[1], max_indicies // heatmap_shape[1]], dim=-1).to(images.device)

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:,
                                                            :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:,
                                                            :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate (cpu)
        keypoints_2d_np = keypoints_2d.detach().cpu().numpy()
        proj_matricies_np = proj_matricies.detach().cpu().numpy()

        keypoints_3d = np.zeros((batch_size, n_joints, 3))
        confidences = np.zeros((batch_size, n_views, n_joints))  # plug
        for batch_i in range(batch_size):
            for joint_i in range(n_joints):
                current_proj_matricies = proj_matricies_np[batch_i]
                points = keypoints_2d_np[batch_i, :, joint_i]
                keypoint_3d, _ = self.triangulate_ransac(
                    current_proj_matricies, points, direct_optimization=self.direct_optimization)
                keypoints_3d[batch_i, joint_i] = keypoint_3d

        keypoints_3d = torch.from_numpy(keypoints_3d).type(
            torch.float).to(images.device)
        confidences = torch.from_numpy(confidences).type(
            torch.float).to(images.device)

        return keypoints_3d, keypoints_2d, heatmaps, confidences

    def triangulate_ransac(self, proj_matricies, points, n_iters=10, reprojection_error_epsilon=15, direct_optimization=True):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2))

            keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(
                proj_matricies[sampled_views], points[sampled_views])
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(
                np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]
                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(
            inlier_proj_matricies, inlier_points)
        reprojection_error_vector = multiview.calc_reprojection_error_matrix(
            np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = multiview.calc_reprojection_error_matrix(
                    np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0,
                                loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(
                np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list


class AlgebraicTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.use_confidences = config.model.use_confidences

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True

        self.backbone = pose_resnet.get_pose_net(
            config.model.backbone, device=device)

        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

    def forward(self, images, proj_matricies, batch, gteval=False, keypoints_2d_gt=None):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integral
        if self.use_confidences:
            heatmaps, _, alg_confidences, _ ,_,_,_= self.backbone(images)
        else:
            heatmaps, _, _, _ = self.backbone(images)
            alg_confidences = torch.ones(
                batch_size * n_views, heatmaps.shape[1]).type(torch.float).to(device)

        heatmaps_before_softmax = heatmaps.view(
            batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d, heatmaps = op.integrate_tensor_2d(
            heatmaps * self.heatmap_multiplier, self.heatmap_softmax)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d = keypoints_2d.view(
            batch_size, n_views, *keypoints_2d.shape[1:])
        alg_confidences = alg_confidences.view(
            batch_size, n_views, *alg_confidences.shape[1:])

        # norm confidences
        alg_confidences = alg_confidences / \
            alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(
            heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:,
                                                            :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:,
                                                            :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        if gteval and keypoints_2d_gt is not None:
            keypoints_2d = keypoints_2d_gt

        # triangulate
        try:
            keypoints_3d = multiview.triangulate_batch_of_points(
                proj_matricies, keypoints_2d,
                confidences_batch=alg_confidences
            )
        except RuntimeError as e:
            print("Error: ", e)

            print("confidences =", confidences_batch_pred)
            print("proj_matricies = ", proj_matricies)
            print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
            exit()

        return keypoints_3d, keypoints_2d, heatmaps, alg_confidences


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(
            config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(
            config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)

    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(
                batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(
            images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / \
                vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(
                    image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0)
                                     for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(
            batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array(
                [self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(
                self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + \
                (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + \
                (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + \
                (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(
                self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(
                coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(
                    coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes,
                                        volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(
            volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points


class StereoTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_generation_method = config.model.volume_generation_method
        self.depth_caculation_method = config.model.depth_caculation_method
        self.volume_net_type = config.model.volume_net_type
        self.probability_propagate = config.model.probability_propagate

        # stereo volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size
        self.min_dis_o = config.model.min_dis
        self.max_dis_o = config.model.max_dis
        self.min_dis = config.model.min_dis
        self.max_dis = config.model.max_dis
        self.feature_layer_idx = config.model.feature_layer_idx
        self.feature_layers = len(self.feature_layer_idx)

        self.kind = config.model.kind
        self.dataset_kind = config.dataset.kind

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(
            config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        self.backbone = pose_resnet.get_pose_net(
            config.model.backbone, device=device)

        if config.model.volume_net_type != "VHonly" and config.model.volume_net_type != "V2V_mask":
            for p in self.backbone.final_layer.parameters():
                p.requires_grad = False

        if config.model.volume_net_type == "V2V":
            self.process_features = nn.Sequential(
                nn.Conv2d(256, 32, 1)
            )
            self.process_features_2 = nn.Sequential(
                nn.Conv2d(32, 16, 1)
            )
            self.volume_net = V2VModel(
                32, self.num_joints, layers=config.model.encoder_layers)

        elif config.model.volume_net_type == "V2V_mask":
            self.process_features = nn.Sequential(
                nn.Conv2d(256, 32, 1)
            )
            self.process_features_2 = nn.Sequential(
                nn.Conv2d(32, 16, 1)
            )
            self.volume_net = V2VModel(
                32, self.num_joints, layers=config.model.encoder_layers,if_conf=config.model.stereo_confidence)

        elif config.model.volume_net_type == "VH":
            self.process_features = nn.Sequential(
                nn.Conv2d(256, 128, 1)
            )
            self.volume_net = VHModel(128*2, self.num_joints)

        elif config.model.volume_net_type == "VHonly":
            self.volume_net = VHModel(2, 1)

        elif config.model.volume_net_type == "V2V_c2f":
            # self.process_features_1 = nn.Sequential(
            #     nn.Conv2d(512, 256, 1)
            # )
            self.process_features = nn.Sequential(
                nn.Conv2d(256, 32, 1)
            )
            self.process_features_2 = nn.Sequential(
                nn.Conv2d(32, 16, 1)
            )
            self.volume_net = V2VModel(
                32, self.num_joints, out_sep=True, layers=config.model.encoder_layers, if_conf=config.model.stereo_confidence)
            
            self.output_net = []
            for i in range(self.num_joints):
                self.output_net.append(
                    nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0))
            self.output_net = nn.Sequential(*self.output_net)
            if config.model.train_module == "refine":
                for p in self.volume_net.parameters():
                    p.requires_grad = False
                for p in self.output_net.parameters():
                    p.requires_grad = False
            if self.feature_layers >= 2:
                self.volume_net_2 = V2VModel(
                    32, self.num_joints, out_sep=True, layers=config.model.encoder_layers)
                self.output_net_2 = []
                for i in range(self.num_joints):
                    self.output_net_2.append(
                        nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0))
                self.output_net_2 = nn.Sequential(*self.output_net_2)

            zzz, yyy, xxx = torch.meshgrid(torch.arange(0, 4, device=device), torch.arange(
                4, device=device), torch.arange(4, device=device))
            grid = torch.stack([zzz, yyy, xxx], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))
            self.coord_volume_fine = grid.reshape(
                4, 4, 4, 3)

        # stereo_volume coordinate
        zzz, yyy, xxx = torch.meshgrid(torch.arange(self.min_dis, self.max_dis+1, device=device), torch.arange(
            self.volume_size, device=device), torch.arange(self.volume_size, device=device))
        grid = torch.stack([zzz, yyy, xxx], dim=-1).type(torch.float)
        grid = grid.reshape((-1, 3))
        self.coord_volume = grid.reshape(
            self.max_dis-self.min_dis+1, self.volume_size, self.volume_size, 3)

        # 3D_volume coordinate
        self.coord_volume_3d_fin = None
        if self.probability_propagate:
            config_psm = config.model.psm
            self.cuboid_side = config_psm.cuboid_size
            self.volume_size_3d_init = config_psm.volume_size_init
            self.volume_size_3d_fin = config_psm.volume_size_fin
            self.volume_3d_multiplier = config_psm.volume_multiplier
            self.heatmap_gaussion_std = config_psm.heatmap_gaussion_std
            self.volume_3d_summax = config_psm.volume_summax

            sides = np.array(
                [self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = - sides / 2

            x, y, z = torch.meshgrid(torch.arange(self.volume_size_3d_fin, device=device), torch.arange(
                self.volume_size_3d_fin, device=device), torch.arange(self.volume_size_3d_fin, device=device))
            grid_3d_fin = torch.stack([x, y, z], dim=-1).type(torch.float)
            grid_3d_fin = grid_3d_fin.reshape((-1, 3))

            grid_coord_fin = torch.zeros_like(grid_3d_fin)
            grid_coord_fin[:, 0] = position[0] + (sides[0] /
                                                  (self.volume_size_3d_fin - 1)) * grid_3d_fin[:, 0]
            grid_coord_fin[:, 1] = position[1] + (sides[1] /
                                                  (self.volume_size_3d_fin - 1)) * grid_3d_fin[:, 1]
            grid_coord_fin[:, 2] = position[2] + (sides[2] /
                                                  (self.volume_size_3d_fin - 1)) * grid_3d_fin[:, 2]
            self.coord_volume_3d_fin = grid_coord_fin.reshape(self.volume_size_3d_fin,
                                                              self.volume_size_3d_fin,
                                                              self.volume_size_3d_fin, 3)

            with open(config_psm.skeleton_path, 'r') as f:
                datas = json.load(f)
            skeleton_data = eval(datas)

            self.psm_gnn = PSMGNNModel_nolearn(
                self.num_joints, self.num_joints, skeleton_data, config_psm.data_dir, device, config_psm)

    def stereo_volume_generation(self, left_feature, right_feature, min_dis, max_dis, mode="coarse"):
        b, c, h, w = left_feature.shape
        dis_size = max_dis - min_dis + 1
        left_size = abs(min_dis)
        right_size = max_dis

        if self.volume_generation_method == 'concat':
            cost_volume = left_feature.new_zeros(b, 2 * c, dis_size, h, w)
            if mode == "coarse":
                for i in range(dis_size):
                    if i < left_size:
                        cost_volume[:, :, i, :, :(w-left_size + i)] = torch.cat((left_feature[:, :, :, :(w-left_size + i)], right_feature[:, :, :, (left_size-i):]),
                                                                                dim=1)
                    elif i == left_size:
                        cost_volume[:, :, i, :, :] = torch.cat(
                            (left_feature, right_feature), dim=1)
                    else:
                        cost_volume[:, :, i, :, (i-left_size):] = torch.cat((left_feature[:, :, :, (i-left_size):], right_feature[:, :, :, :-(i-left_size)]),
                                                                            dim=1)
            else:
                w_r = right_feature.shape[-1]
                right_ori = w_r - w
                for i in range(dis_size):
                    if i == 0:
                        cost_volume[:, :, i, :, :] = torch.cat(
                            (left_feature, right_feature[:, :, :, right_ori:]), dim=1)
                    else:
                        cost_volume[:, :, i, :, :] = torch.cat(
                            (left_feature, right_feature[:, :, :, (right_ori-i):(-i)]), dim=1)

        else:
            raise NotImplementedError

        # [B, C, D, H, W] or [B, D, H, W]
        cost_volume = cost_volume.contiguous()

        return cost_volume

    def forward(self, images, K, T, R, t, proj_matricies, batch, occlusion, keypoints_2d_gt=None, gteval=False, keypoints_3d_gt=None, keypoints_3d_gt_1=None):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        # heatmaps, features_list, _, vol_confidences = self.backbone(
        #     images, self.feature_layer_idx)
        heatmaps, features_list, _, vol_confidences, x_1, x_2, x_3= self.backbone(
            images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        x_1 = x_1.view(batch_size, n_views,  *x_1.shape[1:]) 
        if x_2 is not None:     
            x_2 = x_2.view(batch_size, n_views,  *x_2.shape[1:]) 
            x_3 = x_3.view(batch_size, n_views,  *x_3.shape[1:]) 
        # features = features.view(batch_size, n_views, *features.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(
            images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # process features before concatenating
        keypoints_2d_list = []
        keypoints_2d_low_list = []
        if self.volume_net_type == "V2V_c2f" and self.feature_layer_idx[0] == 0:
            features[0] = self.process_features_1(features[0])
        for i in range(self.feature_layers):
            features = features_list[i]
            features = self.process_features(features)
            if "V2V" in self.volume_net_type:
                features = self.process_features_2(features)
            features = features.view(batch_size, n_views, *features.shape[1:])

            if self.volume_net_type == "V2V_mask":
                # heatmaps = nn.functional.softmax(heatmaps, dim=2)

                heatmaps = heatmaps.reshape(batch_size, n_views, self.num_joints, -1)
                heatmaps = nn.functional.softmax(heatmaps, dim=3)
                heatmaps = heatmaps.reshape(batch_size, n_views, self.num_joints, heatmap_shape[0], heatmap_shape[1])
                heatmaps = torch.sum(heatmaps, dim=2).unsqueeze(2)
                x_1 = nn.functional.softmax(x_1, dim=2)
                x_1 = torch.sum(x_1, dim=2)
                if x_2 is not None:  
                    x_2 = nn.functional.softmax(x_2, dim=2)
                    x_2 = torch.sum(x_2, dim=2)
                    x_3 = nn.functional.softmax(x_3, dim=2)
                    x_3 = torch.sum(x_3, dim=3)
                features = features * heatmaps
            # calcualte shapes
            features_shape = tuple(features.shape[2:])

            if vol_confidences is not None:
                vol_confidences = vol_confidences.view(
                    batch_size, n_views, *vol_confidences.shape[1:])
                
            if i == 0:
                left_features = features[:, 0, :]
                right_features = features[:, 1, :]
                volumes = self.stereo_volume_generation(
                    left_features, right_features, self.min_dis, self.max_dis)

                self.min_dis = 0
                self.max_dis = 3
            else:
                volumes = torch.empty(
                    (batch_size, n_joints, 2 * features_shape[0], (self.max_dis - self.min_dis+1), 4, 4)).to(device)

                keypoints_2d_low_np = (keypoints_2d_low_list[i-1] * (
                    features_shape[1]/image_shape[0])).detach().cpu().numpy().astype(np.int8)
                for joint_i in range(n_joints):
                    left_features = torch.zeros(
                        (batch_size, features_shape[0], 4, 4)).to(device)
                    right_features = torch.zeros(
                        (batch_size, features_shape[0], 4, 8)).to(device)

                    left_feature_start_x = keypoints_2d_low_np[:,
                                                               0, joint_i, 0]
                    left_feature_end_x = left_feature_start_x + 4
                    left_feature_start_y = keypoints_2d_low_np[:,
                                                               0, joint_i, 1]
                    left_feature_end_y = left_feature_start_y + 4
                    right_feature_start_x = keypoints_2d_low_np[:,
                                                                1, joint_i, 0] - 4
                    right_feature_end_x = keypoints_2d_low_np[:,
                                                              1, joint_i, 0] + 4
                    right_feature_start_y = keypoints_2d_low_np[:,
                                                                1, joint_i, 1]
                    right_feature_end_y = left_feature_start_y + 4
                    for batch_i in range(batch_size):
                        left_features[batch_i] = features[batch_i, 0, :, left_feature_start_y[batch_i]
                            : left_feature_end_y[batch_i], left_feature_start_x[batch_i]: left_feature_end_x[batch_i]]
                        right_start_x = right_feature_start_x[batch_i]
                        right_end_x = right_feature_end_x[batch_i]
                        right_start_y = right_feature_start_y[batch_i]
                        right_end_y = right_feature_end_y[batch_i]
                        if right_start_x >= 0 and right_end_x < features_shape[2] and right_start_y >= 0 and right_end_y < features_shape[1]:
                            right_features[batch_i] = features[batch_i, 1, :,
                                                               right_start_y: right_end_y, right_start_x:right_end_x]
                        else:
                            x_0 = 0
                            for x_i in range(right_start_x, right_end_x):
                                if x_i < 0 or x_i >= features_shape[2]:
                                    x_0 += 1
                                    continue
                                y_0 = 0
                                for y_i in range(right_start_y, right_end_y):
                                    if y_i < 0 or y_i >= features_shape[1]:
                                        y_0 += 1
                                        continue
                                    right_features[batch_i, :, y_0,
                                                   x_0] = features[batch_i, 1, :, y_i, x_i]

                    volumes[:, joint_i] = self.stereo_volume_generation(
                        left_features, right_features, self.min_dis, self.max_dis, mode="fine")

                volumes = volumes.view(
                    batch_size * n_joints, 2 * features_shape[0], (self.max_dis - self.min_dis+1), 4, 4)

            if i == self.feature_layers-1:
                self.min_dis = self.min_dis_o
                self.max_dis = self.max_dis_o

            # integral 3d
            if i == 0:
                volumes, stereo_conf = self.volume_net(volumes)
            if i == 1:
                volumes = self.volume_net_2(volumes)
            if self.volume_net_type == "V2V_c2f":
                volumes_joint = torch.zeros(
                    (batch_size, n_joints, *volumes.shape[-3:])).to(device)
                if i == 0:
                    for joint_i in range(n_joints):
                        volumes_joint[:, joint_i] = self.output_net[joint_i](
                            volumes).squeeze(1)
                if i == 1:
                    volumes = volumes.reshape(
                        batch_size, n_joints, -1, 4, 4, 4)
                    for joint_i in range(n_joints):
                        volumes_joint[:, joint_i] = self.output_net_2[joint_i](
                            volumes[:, joint_i]).squeeze(1)
            else:
                volumes_joint = volumes

            volumes_joint_ori = volumes_joint.clone()
            if i == 0:
                vol_keypoints_25d, volumes_joint = op.integrate_tensor_3d_with_coordinates(
                    volumes_joint * self.volume_multiplier, self.coord_volume, softmax=self.volume_softmax)
            else:
                vol_keypoints_25d, volumes_joint = op.integrate_tensor_3d_with_coordinates(
                    volumes_joint * self.volume_multiplier, self.coord_volume_fine, softmax=self.volume_softmax)

            # upscale keypoints_2.5d, because volume shape != image shape
            keypoints_25d_transformed = vol_keypoints_25d.clone()
            vol_keypoints_25d_ori = vol_keypoints_25d.clone()
            keypoints_25d_transformed[:, :, 0] = vol_keypoints_25d[:,
                                                                   :, 0] * (image_shape[1] / features_shape[2])
            keypoints_25d_transformed[:, :, 1] = vol_keypoints_25d[:,
                                                                   :, 1] * (image_shape[0] / features_shape[1])
            keypoints_25d_transformed[:, :, 2] = vol_keypoints_25d[:,
                                                                   :, 2] * (image_shape[1] / features_shape[2])
            vol_keypoints_25d = keypoints_25d_transformed
      
            if gteval and keypoints_2d_gt is not None:
                vol_keypoints_25d[:, :, 0] = keypoints_2d_gt[:,
                                                             0, :, 0] - keypoints_2d_gt[:, 1, :, 0]
                vol_keypoints_25d[:, :, 1] = keypoints_2d_gt[:, 0, :, 1]
                vol_keypoints_25d[:, :, 2] = keypoints_2d_gt[:, 0, :, 0]

            

            if i == 0:
                keypoints_2d = torch.zeros(
                    (batch_size, n_views, n_joints, 2)).to(device)
                keypoints_2d[:, 0, :, 0] = vol_keypoints_25d[:, :, 2]
                keypoints_2d[:, 0, :, 1] = vol_keypoints_25d[:, :, 1]
                keypoints_2d[:, 1, :, 0] = vol_keypoints_25d[:,
                                                             :, 2] - vol_keypoints_25d[:, :, 0]
                keypoints_2d[:, 1, :, 1] = vol_keypoints_25d[:, :, 1]
                keypoints_2d_list.append(keypoints_2d)

            else:
                keypoints_2d = torch.zeros(
                    (batch_size, n_views, n_joints, 2)).to(device)
                keypoints_2d_low = keypoints_2d_low_list[i-1]
                keypoints_2d[:, 0, :, 0] = keypoints_2d_low[:, 0,
                                                            :, 0]+vol_keypoints_25d[:, :, 2]
                keypoints_2d[:, 0, :, 1] = keypoints_2d_low[:, 0,
                                                            :, 1]+vol_keypoints_25d[:, :, 1]
                keypoints_2d[:, 1, :, 0] = keypoints_2d[:,
                                                        0, :, 0] - (keypoints_2d_low[:,
                                                                                     0, :, 0] - keypoints_2d_low[:,
                                                                                                                 1, :, 0] + vol_keypoints_25d[:, :, 0])
                keypoints_2d[:, 1, :, 1] = keypoints_2d_low[:, 1,
                                                            :, 1]+vol_keypoints_25d[:, :, 1]
                keypoints_2d_list.append(keypoints_2d)

            with torch.no_grad():
                vol_keypoints_25d_ori = torch.floor(vol_keypoints_25d_ori)
                vol_keypoints_25d_ori[:, :, 0] = vol_keypoints_25d_ori[:,
                                                                       :, 0] * (image_shape[1] / features_shape[2])
                vol_keypoints_25d_ori[:, :, 1] = vol_keypoints_25d_ori[:,
                                                                       :, 1] * (image_shape[0] / features_shape[1])
                vol_keypoints_25d_ori[:, :, 2] = vol_keypoints_25d_ori[:,
                                                                       :, 2] * (image_shape[1] / features_shape[2])

                keypoints_2d_low = torch.zeros(
                    (batch_size, n_views, n_joints, 2)).to(device)
                if i == 0:
                    keypoints_2d_low[:, 0, :,
                                     0] = vol_keypoints_25d_ori[:, :, 2]
                    keypoints_2d_low[:, 0, :,
                                     1] = vol_keypoints_25d_ori[:, :, 1]
                    keypoints_2d_low[:, 1, :, 0] = vol_keypoints_25d_ori[:,
                                                                         :, 2] - vol_keypoints_25d_ori[:, :, 0]
                    keypoints_2d_low[:, 1, :,
                                     1] = vol_keypoints_25d_ori[:, :, 1]
                else:
                    keypoints_2d_low_before = keypoints_2d_low_list[i-1]
                    keypoints_2d_low[:, 0, :,
                                     0] = keypoints_2d_low_before[:, 0, :,
                                                                  0] + vol_keypoints_25d_ori[:, :, 2]
                    keypoints_2d_low[:, 0, :,
                                     1] = keypoints_2d_low_before[:, 0, :,
                                                                  1] + vol_keypoints_25d_ori[:, :, 1]
                    keypoints_2d_low[:, 1, :, 0] = keypoints_2d_low[:, 0, :,
                                                                    0] - vol_keypoints_25d_ori[:, :, 0]

                    keypoints_2d_low[:, 1, :,
                                     1] = keypoints_2d_low_before[:, 1, :,
                                                                  1]+vol_keypoints_25d_ori[:, :, 1]
                keypoints_2d_low_list.append(keypoints_2d_low)

        if self.depth_caculation_method == 'tri':

            # triangulate
            # if self.dataset_kind == "mhad":
            #     keypoints_2d_list[-1][:,:,6,1] = keypoints_2d_list[-1][:,:,6,1]+10
            try:
                vol_keypoints_3d_w = multiview.triangulate_batch_of_points(
                    proj_matricies, keypoints_2d_list[-1]
                )
            except RuntimeError as e:
                print("Error: ", e)
                print("proj_matricies = ", proj_matricies)
                print("keypoints_2d_batch_pred =", keypoints_2d)
                exit()

        else:
            # caculate the baseline/disparity = depth/f_x
            K_l = K[:, 0]
            K_r = K[:, 1]
            T_l = T[:, 0]
            T_r = T[:, 1]
            baselines = torch.sqrt(torch.sum((T_l-T_r)**2, dim=1))
            dis_left = vol_keypoints_25d[:, :, 2] - K_l[:, 0:1, 2]
            dis_right = -vol_keypoints_25d[:, :, 2] + \
                vol_keypoints_25d[:, :, 0] + K_r[:, 0:1, 2]
            dis = dis_left + dis_right * \
                (K_l[:, 0, 0] / K_r[:, 0, 0]).unsqueeze(dim=1)
            ratio_BD = baselines / dis

            # transform from 2.5d to 3d
            vol_keypoints_3d = torch.zeros_like(vol_keypoints_25d)
            vol_keypoints_3d[:, :, 0] = (
                vol_keypoints_25d[:, :, 2] - K_l[:, 0:1, 2]) * ratio_BD
            vol_keypoints_3d[:, :, 1] = (
                vol_keypoints_25d[:, :, 1] - K_l[:, 1:2, 2]) * ratio_BD
            vol_keypoints_3d[:, :, 2] = K_l[:, 0:1, 0] * ratio_BD

            # transform from left camera coordinates to world coordinates
            R_l = R[:, 0, :]
            t_l = t[:, 0, :]
            vol_keypoints_3d_w = torch.zeros_like(vol_keypoints_3d)
            vol_keypoints_3d_w = (R_l.transpose(1, 2).unsqueeze(
                1)) @ ((vol_keypoints_3d - t_l.permute(0, 2, 1)).unsqueeze(-1))
            vol_keypoints_3d_w = vol_keypoints_3d_w.squeeze(-1)

        volumes_joint_3d = None
        volumes_joint_3d_gt = None
        if self.probability_propagate:
            K_l = K[:, 0]
            K_r = K[:, 1]
            T_l = T[:, 0]
            T_r = T[:, 1]
            R_l = R[:, 0, :]
            R_r = R[:, 1, :]
            t_l = t[:, 0, :]
            t_r = t[:, 1, :]
            vol_keypoints_3d_l = R_l.unsqueeze(
                1) @ vol_keypoints_3d_w.unsqueeze(-1) + t_l.unsqueeze(1)
            vol_keypoints_3d_r = R_r.unsqueeze(
                1) @ vol_keypoints_3d_w.unsqueeze(-1) + t_r.unsqueeze(1)
            # center_position = vol_keypoints_3d_l.squeeze(-1)[:, 6, :]
            k_2d_l = (K_l.unsqueeze(1) @ vol_keypoints_3d_l).squeeze(-1)
            k_2d_l[:,:,0] = k_2d_l[:,:,0] / k_2d_l[:,:,2]
            k_2d_l[:,:,1] = k_2d_l[:,:,1] / k_2d_l[:,:,2]
            k_2d_r = (K_r.unsqueeze(1) @ vol_keypoints_3d_r).squeeze(-1)
            k_2d_r[:,:,0] = k_2d_r[:,:,0] / k_2d_r[:,:,2]
            k_2d_r[:,:,1] = k_2d_r[:,:,1] / k_2d_r[:,:,2]
            k_2d_l[:,:,2] = k_2d_l[:,:,0] - k_2d_r[:,:,0]
            center_position = vol_keypoints_3d_w[:, 6, :]

            volumes_joint_3d = op.gaussian_3d_relative_heatmap(
                    vol_keypoints_3d_w, center_position, self.coord_volume_3d_fin, self.heatmap_gaussion_std)

            volumes_3d = self.psm_gnn(volumes_joint_3d, occlusion)
            vol_keypoints_3d_w_3, volumes_3d = op.integrate_tensor_3d_with_coordinates(
                volumes_3d * self.volume_multiplier, self.coord_volume_3d_fin, softmax=False, summax=self.volume_3d_summax)
            vol_keypoints_3d_w_3 = vol_keypoints_3d_w_3 + \
                center_position.reshape(batch_size, 1, -1)

        return vol_keypoints_3d_w, keypoints_2d_list[-1], heatmaps, volumes_joint, self.coord_volume, stereo_conf, x_1, x_2, x_3
