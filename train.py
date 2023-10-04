import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from lib.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet, StereoTriangulationNet
from lib.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss, Mpjpe, HeatmapCELoss
from lib.models.mask_transofrmer import Masked_ST
from lib.utils import img, multiview, op, vis, misc, cfg
from lib.datasets import mhad_stereo, human36m
from lib.datasets import utils as dataset_utils
from thop import profile


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True,
                        help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true',
                        help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val',
                        help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int,
                        help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr",
                        help="Path, where logs will be stored")

    args = parser.parse_args()
    return args

def setup_h36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(
                config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(
                config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.h36m_labels_path,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            crop=config.dataset.train.crop if hasattr(
                config.dataset.train, "crop") else True,
            dataset="h36m",
            rectificated=True,
            withsilhouette=False
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (
                train_sampler is None),  # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=False
        )

    # val
    val_dataset = human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.train.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(
            config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(
            config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.h36m_labels_path,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        crop=config.dataset.val.crop if hasattr(
            config.dataset.val, "crop") else True,
        dataset="h36m",
        rectificated=True,
        withsilhouette=False
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(
            config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=False
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_mhad_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = mhad_stereo.MHADStereoViewDataset(
            mhad_root=config.dataset.train.mhad_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(
                config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(
                config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            scale_bbox=config.dataset.train.scale_bbox,
            norm_image = config.dataset.norm if hasattr(config.dataset, 'norm') else True,
            kind=config.kind,
            crop=config.dataset.train.crop if hasattr(
                config.dataset.train, "crop") else True,
            dataset="mhad",
            rectificated=True,
            baseline_width=config.dataset.train.baseline_width if hasattr(
                config.dataset.train, "baseline_width") else 'm'
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (
                train_sampler is None),  # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=False
        )

    # val
    val_dataset = mhad_stereo.MHADStereoViewDataset(
        mhad_root=config.dataset.train.mhad_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(
            config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(
            config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        crop=config.dataset.val.crop if hasattr(
            config.dataset.val, "crop") else True,
        dataset="mhad",
        rectificated=True,
        baseline_width=config.dataset.val.baseline_width if hasattr(
            config.dataset.val, "baseline_width") else 'm'
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(
            config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=False
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'mhad':
        train_dataloader, val_dataloader, train_sampler = setup_mhad_dataloaders(
            config, is_train, distributed_train)
    elif config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_h36m_dataloaders(
            config, is_train, distributed_train)
    else:
        raise NotImplementedError(
            "Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title,
                                     datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def one_epoch(model, criterion, opt, config, dataloader, device, epoch, refine_model=None, opt_refine=None, n_iters_total=0, is_train=True, caption='', master=False, experiment_dir=None, writer=None, visible=False):
    name = "train" if is_train else "val"
    model_type = config.model.name

    if is_train:
        model.train()
        if refine_model is not None:
            refine_model.train()
    else:
        model.eval()
        if refine_model is not None:
            refine_model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        iterator = enumerate(dataloader)
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        for iter_i, batch in iterator:
            with autograd.detect_anomaly():
                # measure data loading time
                data_time = time.time() - end

                if batch is None:
                    print("Found None batch")
                    continue

                images_batch, keypoints_3d_gt, keypoints_3d_batch_ca, keypoints_3d_validity_gt, keypoints_2d_gt, keypoints_2d_validity_gt, proj_matricies_batch, K_batch, T_batch, R_batch, t_batch, occlusion_left = dataset_utils.prepare_batch(
                    batch, device, config)

                keypoints_2d_pred, cuboids_pred, base_points_pred, confidences_pred = None, None, None, None
                if model_type == "alg" or model_type == "ransac":
                    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
                        images_batch, proj_matricies_batch, batch)
                elif model_type == "vol":
                    keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(
                        images_batch, proj_matricies_batch, batch)
                elif model_type == "stereo":
                    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, volumes_pred, coord_volumes_pred, stereo_conf, mask_1, mask_2, mask_3 = model(
                        images_batch, K_batch, T_batch, R_batch, t_batch, proj_matricies_batch, batch, batch['occlusion'], keypoints_2d_gt=keypoints_2d_gt, gteval=False, keypoints_3d_gt=keypoints_3d_batch_ca[:, 0], keypoints_3d_gt_1=keypoints_3d_gt)
                    # macs, params = profile(model, inputs=(images_batch, K_batch, T_batch, R_batch, t_batch, proj_matricies_batch, batch))
                    # print("MAC:{}, PAR:{}".format(macs, params))

                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(
                    images_batch.shape[3:])
                n_joints = keypoints_3d_pred.shape[1]

                keypoints_3d_binary_validity_gt = (
                    keypoints_3d_validity_gt > 0.0).type(torch.float32)

                scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(
                    config.opt, "scale_keypoints_3d") else 1.0

                # 1-view case
                if n_views == 1:
                    if config.kind == "human36m":
                        base_joint = 6
                    elif config.kind == "coco":
                        base_joint = 11

                    keypoints_3d_gt_transformed = keypoints_3d_gt.clone()
                    keypoints_3d_gt_transformed[:, torch.arange(
                        n_joints) != base_joint] -= keypoints_3d_gt_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_gt = keypoints_3d_gt_transformed

                    keypoints_3d_pred_transformed = keypoints_3d_pred.clone()
                    keypoints_3d_pred_transformed[:, torch.arange(
                        n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_pred = keypoints_3d_pred_transformed

                if (refine_model is not None) and config.model.transformer_refine.relative_pose:
                    keypoints_root = keypoints_3d_pred[:, 6:7, :].clone()
                    keypoints_3d_pred = keypoints_3d_pred - keypoints_root

                keypoints_3d_pred_refined = keypoints_3d_pred
                if refine_model is not None:
                    if config.model.transformer_refine.if_mask:
                        masked_joints = torch.tensor(
                            [0, 10000, 0]).float().cuda().to(device)
                        keypoints_3d_pred = torch.where(
                            occlusion_left == 1, masked_joints, keypoints_3d_pred)
                        keypoints_3d_pred_refined_0, attention = refine_model(
                            keypoints_3d_pred, stereo_conf)
                        with torch.no_grad():
                            joints_confidence = torch.sum(attention, dim=(1, 2)).reshape(batch_size, -1, 1)
                            masked_conf = torch.zeros_like(joints_confidence)
                            masked_confidence = torch.where(
                                occlusion_left == 1, joints_confidence, masked_conf)
                            _, topk_index = torch.topk(
                                masked_confidence, 2, dim=1)
                            for batch_i in range(batch_size):
                                occlusion_left[batch_i, topk_index[batch_i]] = 0
                        keypoints_3d_pred = torch.where(
                            occlusion_left == 1, masked_joints, keypoints_3d_pred_refined_0)
                    
                    keypoints_3d_pred_refined, attns = refine_model(
                        keypoints_3d_pred, stereo_conf)
                    if config.model.transformer_refine.iterative:
                        keypoints_3d_pred = torch.where(
                            occlusion_left == 1, keypoints_3d_pred_refined, keypoints_3d_pred)
                        keypoints_3d_pred_refined, attns = refine_model(
                            keypoints_3d_pred, stereo_conf)

                if (refine_model is not None):
                    if hasattr(config.model.transformer_refine, "body_only") and config.model.transformer_refine.body_only:
                        keypoints_3d_pred_refined[:, [0,5,9,10,15]] = keypoints_3d_pred[:, [0,5,9,10,15]]
                    if config.model.transformer_refine.relative_pose:
                        keypoints_3d_pred_refined = keypoints_3d_pred_refined + keypoints_root
                   
                # calculate loss
                total_loss = 0.0
                if config.model.backbone.retrainwith2d:
                    keypoints_2d_mask = keypoints_2d_gt - \
                        keypoints_2d_low_list[0]
                    zero_mask = torch.zeros_like(keypoints_2d_mask)
                    keypoints_2d_mask = torch.where(
                        keypoints_2d_mask > 16, zero_mask, keypoints_2d_mask)
                    zero_mask = torch.zeros_like(keypoints_2d_mask[:, 0])
                    keypoints_2d_mask[:, 0] = torch.where(
                        keypoints_2d_mask[:, 0] < 0, zero_mask, keypoints_2d_mask[:, 0])
                    keypoints_2d_mask[:, 1] = torch.where(
                        keypoints_2d_mask[:, 1] < -16, zero_mask, keypoints_2d_mask[:, 1])
                    keypoints_2d_validity_idx = torch.nonzero(
                        keypoints_2d_mask == 0)
                    keypoints_2d_validity = keypoints_2d_validity_gt.clone()
                    keypoints_2d_validity = torch.tile(
                        keypoints_2d_validity, (1, 1, 1, 2))
                    keypoints_2d_validity[keypoints_2d_validity_idx[:, 0], keypoints_2d_validity_idx[:,
                                                                                                     1], keypoints_2d_validity_idx[:, 2], keypoints_2d_validity_idx[:, 3]] = 0
                    zero_mask = torch.zeros_like(
                        keypoints_2d_validity[:, :, :, 0])
                    keypoints_2d_validity[:, :, :, 0] = torch.where(
                        keypoints_2d_validity[:, :, :, 1] == 0, zero_mask, keypoints_2d_validity[:, :, :, 0])
                    keypoints_2d_validity[:, :, :, 1] = torch.where(
                        keypoints_2d_validity[:, :, :, 0] == 0, zero_mask, keypoints_2d_validity[:, :, :, 1])
                    zero_mask = torch.zeros_like(
                        keypoints_2d_validity[:, 0, :])
                    keypoints_2d_validity[:, 0, :] = torch.where(
                        keypoints_2d_validity[:, 1, :] == 0, zero_mask, keypoints_2d_validity[:, 0, :])
                    keypoints_2d_validity[:, 1, :] = torch.where(
                        keypoints_2d_validity[:, 0, :] == 0, zero_mask, keypoints_2d_validity[:, 1, :])

                    if config.model.train_module == "refine":
                        # filter totally bad corase results
                        loss = criterion(
                            keypoints_2d_pred, keypoints_2d_gt, keypoints_2d_validity)
                        total_loss += loss
                    else:
                        for i, keypoints_2d_pred in enumerate(keypoints_2d_pred_list):
                            if i == 1:
                                keypoints_2d_validity_gt = keypoints_2d_validity
                            loss = criterion(keypoints_2d_pred,
                                             keypoints_2d_gt, keypoints_2d_validity_gt)
                            total_loss += loss
                else:
                    if config.model.probability_propagate:
                        loss = criterion(keypoints_3d_pred_3 * scale_keypoints_3d,
                                         keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                    else:
                        loss = criterion(keypoints_3d_pred_refined,
                                         keypoints_3d_gt, keypoints_3d_binary_validity_gt)
                    total_loss += loss

                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # stereo volume 3dpose loss
                use_stereo_volume_loss = config.model.psm.use_stereo_volume_loss if hasattr(
                    config.model.psm, "use_stereo_volume_loss") else False
                if use_stereo_volume_loss:
                    loss = criterion(keypoints_3d_pred * scale_keypoints_3d,
                                     keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                    weight = config.model.psm.stereo_volume_loss_weight if hasattr(
                        config.model.psm, "stereo_volume_loss_weight") else 1.0

                # 3d volume heatmap loss
                use_3d_volume_loss = config.model.psm.use_3d_volume_loss if hasattr(
                    config.model.psm, "use_3d_volume_loss") else False
                if use_3d_volume_loss:
                    loss = criterion(volumes_3d_pred.reshape(batch_size, n_joints, -1),
                                     volumes_3d_gt.reshape(
                                         batch_size, n_joints, -1),
                                     keypoints_3d_binary_validity_gt)
                    weight = config.model.psm.threed_volume_loss_weight if hasattr(
                        config.model.psm, "threed_volume_loss_weight") else 1.0
                    total_loss += weight * loss

                # # 3d volume 3dpose loss
                # use_3d_volume_loss = config.model.psm.use_3d_volume_loss if hasattr(
                #     config.model.psm, "use_3d_volume_loss") else False
                # if use_3d_volume_loss:
                #     loss = criterion(keypoints_3d_pred_2 * scale_keypoints_3d,
                #                      keypoints_3d_batch_ca[:,0] * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                #     weight = config.model.psm.threed_volume_loss_weight if hasattr(
                #         config.model.psm, "threed_volume_loss_weight") else 1.0
                #     total_loss += weight * loss

                # volumetric ce loss
                use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(
                    config.opt, "use_volumetric_ce_loss") else False
                if use_volumetric_ce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()
                    coord_volumes_gt = keypoints_3d_gt
                    if config.model.name == 'stereo':
                        volumes_gt = torch.zeros_like(keypoints_3d_gt)
                        volumes_gt[:, :, 0] = (keypoints_2d_gt[:, 0, :, 0] - keypoints_2d_gt[:, 1, :, 0])*(
                            config.model.volume_size / config.image_shape[1])
                        volumes_gt[:, :, 1] = keypoints_2d_gt[:, 0, :, 1] * \
                            (config.model.volume_size / config.image_shape[0])
                        volumes_gt[:, :, 2] = keypoints_2d_gt[:, 0, :, 0] * \
                            (config.model.volume_size / config.image_shape[1])
                        coord_volumes_gt = volumes_gt

                    loss = volumetric_ce_criterion(
                        coord_volumes_pred, volumes_pred, coord_volumes_gt, keypoints_3d_binary_validity_gt, model=config.model.name)
                    metric_dict['volumetric_ce_loss'].append(loss.item())

                    weight = config.opt.volumetric_ce_loss_weight if hasattr(
                        config.opt, "volumetric_ce_loss_weight") else 1.0
                    total_loss += weight * loss

                use_heatmap_ce_loss = config.opt.use_heatmap_ce_loss if hasattr(
                    config.opt, "use_heatmap_ce_loss") else False
                if use_heatmap_ce_loss:
                    heatmap_ce_criterion = HeatmapCELoss()
                    loss = heatmap_ce_criterion(keypoints_2d_gt, heatmaps_pred)
                    metric_dict['heatmap_ce_loss'].append(loss.item())

                    weight = config.opt.heatmap_ce_loss_weight if hasattr(
                        config.opt, "heatmap_ce_loss_weight") else 1.0
                    total_loss += weight * loss

                metric_dict['total_loss'].append(total_loss.item())

                if is_train:
                    opt.zero_grad()
                    if refine_model is not None and opt_refine is not None:
                        opt_refine.zero_grad()

                    total_loss.backward()

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.opt.grad_clip / config.opt.lr)

                    # metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(
                    #     filter(lambda x: x[1].requires_grad, model.named_parameters())))

                    opt.step()
                    if refine_model is not None and opt_refine is not None:
                        opt_refine.step()

                # calculate metrics
                if config.model.probability_propagate:
                    l2 = KeypointsL2Loss()(keypoints_3d_pred_3 * scale_keypoints_3d, keypoints_3d_gt *
                                           scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                else:
                    l2 = KeypointsL2Loss()(keypoints_3d_pred_refined * scale_keypoints_3d, keypoints_3d_gt *
                                           scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                metric_dict['l2'].append(l2.item())

                if iter_i % 100 == 0:
                    print("iter{}: Loss:{}, 3DL2loss:{}".format(
                        iter_i, total_loss.item(), l2.item()))

                # base point l2
                if base_points_pred is not None:
                    base_point_l2_list = []
                    for batch_i in range(batch_size):
                        base_point_pred = base_points_pred[batch_i]

                        if config.model.kind == "coco":
                            base_point_gt = (
                                keypoints_3d_gt[batch_i, 11, :3] + keypoints_3d[batch_i, 12, :3]) / 2
                        elif config.model.kind == "mpii":
                            base_point_gt = keypoints_3d_gt[batch_i, 6, :3]

                        base_point_l2_list.append(torch.sqrt(torch.sum(
                            (base_point_pred * scale_keypoints_3d - base_point_gt * scale_keypoints_3d) ** 2)).item())

                    base_point_l2 = 0.0 if len(
                        base_point_l2_list) == 0 else np.mean(base_point_l2_list)
                    metric_dict['base_point_l2'].append(base_point_l2)

                # save answers for evalulation
                if not is_train:
                    if config.model.probability_propagate:
                        results['keypoints_3d'].append(
                            keypoints_3d_pred_3.detach().cpu().numpy())
                    else:
                        results['keypoints_3d'].append(
                            keypoints_3d_pred_refined.detach().cpu().numpy())
                    results['keypoints_3d_gt'].append(
                        keypoints_3d_gt.detach().cpu().numpy())
                    results['keypoints_2d_gt'].append(
                        keypoints_2d_gt.detach().cpu().numpy())
                    if keypoints_2d_pred is not None:
                        results['keypoints_2d_pred'].append(
                            keypoints_2d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])
                    results['occlusion'].append(occlusion_left.detach().cpu().numpy())

                # plot visualization
                if master and visible:
                    # vis attention map
                    for b, occ in enumerate(occlusion_left):
                        occ_idx = torch.nonzero(occ)
                        if len(occ_idx) == 0:
                            continue
                        else:          
                            attns_vis = vis.visualize_attns(
                                images_batch, attns
                            )
                            writer.add_image(
                                f"{name}/attns/{b}_{occ_idx.detach().cpu().numpy()}", attns_vis.transpose(2, 0, 1), global_step=n_iters_total)

                    if n_iters_total % config.vis_freq == 0:  # or total_l2.item() > 500.0:
                        vis_kind = config.kind
                        if (config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False):
                            vis_kind = "coco"

                        for batch_i in range(min(batch_size, config.vis_n_elements)):
                            keypoints_vis = vis.visualize_batch(
                                images_batch, heatmaps_pred, keypoints_2d_gt, proj_matricies_batch,
                                keypoints_3d_gt, keypoints_3d_pred,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                confidences_batch=confidences_pred,
                                batch_index=batch_i, size=5,
                                max_n_cols=10
                            )
                            writer.add_image(
                                f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            heatmaps_vis = vis.visualize_heatmaps(
                                images_batch, heatmaps_pred,
                                kind=vis_kind,
                                batch_index=batch_i, size=5,
                                max_n_rows=10, max_n_cols=10
                            )
                            writer.add_image(
                                f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)
                            
                            masks_vis = vis.visualize_masks(
                                images_batch, mask_1, mask_2, mask_3,
                                kind=vis_kind,
                                batch_index=batch_i, size=5,
                                max_n_rows=10, max_n_cols=10
                            )
                            writer.add_image(
                                f"{name}/masks/{batch_i}", masks_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            # if model_type == "vol":
                            #     volumes_vis = vis.visualize_volumes(
                            #         images_batch, volumes_pred, proj_matricies_batch,
                            #         kind=vis_kind,
                            #         cuboids_batch=cuboids_pred,
                            #         batch_index=batch_i, size=5,
                            #         max_n_rows=1, max_n_cols=16
                            #     )
                            #     writer.add_image(
                            #         f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            # if model_type == "stereo":
                            #     volumes_vis = vis.visualize_volumes_stereo(
                            #         images_batch, volumes_pred,
                            #         kind=vis_kind,
                            #         batch_index=batch_i, size=5,
                            #         max_n_rows=1, max_n_cols=16
                            #     )
                            #     writer.add_image(
                            #         f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            #     volumes_3d_pred_whole = torch.sum(
                            #         volumes_3d_pred, dim=1).unsqueeze(1)
                            #     volumes_3d_vis = vis.visualize_volumes_stereo(
                            #         images_batch, volumes_3d_pred_whole,
                            #         kind=vis_kind,
                            #         batch_index=batch_i, size=5,
                            #         max_n_rows=1, max_n_cols=16
                            #     )
                            #     writer.add_image(
                            #         f"{name}/volumes_3d/{batch_i}", volumes_3d_vis.transpose(2, 0, 1), global_step=n_iters_total)

                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0:
                        for p_name, p in model.named_parameters():
                            try:
                                writer.add_histogram(
                                    p_name, p.clone().cpu().data.numpy(), n_iters_total)
                            except ValueError as e:
                                print(e)
                                print(p_name, p)
                                exit()

                    # dump to tensorboard per-iter loss/metric stats
                    if is_train:
                        for title, value in metric_dict.items():
                            writer.add_scalar(
                                f"{name}/{title}", value[-1], n_iters_total)

                    # measure elapsed time
                    batch_time = time.time() - end
                    end = time.time()

                    # dump to tensorboard per-iter time stats
                    writer.add_scalar(f"{name}/batch_time",
                                      batch_time, n_iters_total)
                    writer.add_scalar(f"{name}/data_time",
                                      data_time, n_iters_total)

                    # dump to tensorboard per-iter stats about sizes
                    writer.add_scalar(f"{name}/batch_size",
                                      batch_size, n_iters_total)
                    writer.add_scalar(f"{name}/n_views",
                                      n_views, n_iters_total)

                n_iters_total += 1

    # calculate evaluation metrics
    scalar_metric, scalar_metric_abs = None, None
    if master:
        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

        if not is_train:
            results['keypoints_3d'] = np.concatenate(
                results['keypoints_3d'], axis=0)
            np.save(os.path.join(experiment_dir, "keypoints_3d_pred.npy"),
                    results['keypoints_3d'])
            results['keypoints_3d_gt'] = np.concatenate(
                results['keypoints_3d_gt'], axis=0)
            np.save(os.path.join(experiment_dir, "keypoints_3d_gt.npy"),
                    results['keypoints_3d_gt'])
            results['keypoints_2d_gt'] = np.concatenate(results['keypoints_2d_gt'],
                                                        axis=0)
            np.save(os.path.join(experiment_dir, "keypoints_2d_gt.npy"),
                    results['keypoints_2d_gt'])
            if keypoints_2d_pred is not None:
                results['keypoints_2d_pred'] = np.concatenate(results['keypoints_2d_pred'],
                                                              axis=0)
                np.save(os.path.join(experiment_dir, "keypoints_2d_pred.npy"),
                        results['keypoints_2d_pred'])
            results['indexes'] = np.concatenate(results['indexes'])
            np.save(os.path.join(experiment_dir, "indexes.npy"),
                    results['indexes'])
            results['occlusion'] = np.concatenate(results['occlusion'])
            np.save(os.path.join(experiment_dir, "occlusion.npy"),
                    results['occlusion'])

            try:
                scalar_metric, scalar_metric_abs, full_metric = dataloader.dataset.evaluate(
                    results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, scalar_metric_abs, full_metric = 0.0, 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)

            checkpoint_dir = os.path.join(
                experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            try:
                jdr_2d, jdr_avg = dataloader.dataset.JDR_2d(
                    results['keypoints_2d_pred'], results['keypoints_2d_gt'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                jdr_2d, jdr_avg = {}, 0.0

            metric_dict['dataset_jdr'].append(jdr_avg)
            metric_dict['dataset_metric'].append(scalar_metric)

            # dump full metric
            with open(os.path.join(checkpoint_dir, "jdr_2d.json".format(epoch)), 'w') as fout:
                json.dump(jdr_2d, fout, indent=4, sort_keys=True)
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

        
    return n_iters_total, scalar_metric, scalar_metric_abs


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)

    # config
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = None
    if hasattr(config.opt, 'n_objects_per_epoch'):
        config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet,
        "stereo": StereoTriangulationNet
    }[config.model.name](config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint,
                                map_location=torch.device('cpu'))
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            # if "final_layer" in new_key:
            #         continue
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pretrained weights for whole model")

    # refine_transformer
    transformer_refine = config.model.if_transofrmer_refine

    refine_model_train = None
    refine_model_eval = None
    if transformer_refine:
        refine_model_train = Masked_ST(num_joints=config.model.backbone.num_joints, in_chans=3,
                                       embed_dim_ratio=config.model.transformer_refine.embed_dim,
                                       depth=config.model.transformer_refine.depth_nums,
                                       num_heads=config.model.transformer_refine.head_nums,
                                       mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1).cuda().to(device)
        refine_model_eval = Masked_ST(num_joints=config.model.backbone.num_joints, in_chans=3,
                                      embed_dim_ratio=config.model.transformer_refine.embed_dim,
                                      depth=config.model.transformer_refine.depth_nums,
                                      num_heads=config.model.transformer_refine.head_nums,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0).cuda().to(device)
        if config.model.transformer_refine.init_weights:
            # state_dict = torch.load(config.model.checkpoint)
            state_dict = torch.load(config.model.transformer_refine.checkpoint,
                                    map_location=torch.device('cpu'))
            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)

            refine_model_train.load_state_dict(state_dict, strict=False)
            refine_model_eval.load_state_dict(state_dict, strict=False)
            # model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained weights for refine transofrmer model")

    # criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss,
        'MPJPE': Mpjpe
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer
    opt = None
    opt_step = None
    opt_refine = None
    opt_refine_step = None
    if not args.eval:
        if config.model.name == "vol":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(
                     config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(
                     config.opt, "volume_net_lr") else config.opt.lr}
                 ],
                lr=config.opt.lr
            )
        if config.model.name == "stereo":
            if config.model.volume_net_type == "V2V_c2f":
                opt = torch.optim.Adam(
                    [
                        {'params': model.backbone.parameters(), 'lr': config.opt.backbone_lr if hasattr(
                            config.opt, "backbone_lr") else config.opt.lr},
                        {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(
                         config.opt, "process_features_lr") else config.opt.lr},
                        {'params': model.process_features_2.parameters(), 'lr': config.opt.process_features_2_lr if hasattr(
                            config.opt, "process_features_2_lr") else config.opt.lr},
                        {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(
                            config.opt, "volume_net_lr") else config.opt.lr},
                        {'params': model.output_net.parameters(), 'lr': config.opt.output_net_lr if hasattr(
                            config.opt, "output_net_lr") else config.opt.lr},
                        {'params': model.volume_net_2.parameters(), 'lr': config.opt.volume_net_2_lr if hasattr(
                            config.opt, "volume_net_2_lr") else config.opt.lr},
                        {'params': model.output_net_2.parameters(), 'lr': config.opt.output_net_2_lr if hasattr(
                            config.opt, "output_net_2_lr") else config.opt.lr}

                    ],
                    lr=config.opt.lr
                )
            else:
                opt = torch.optim.Adam(
                    [{'params': model.backbone.parameters(), 'lr': config.opt.backbone_lr if hasattr(config.opt, "backbone_lr") else config.opt.lr},
                     {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(
                         config.opt, "volume_net_lr") else config.opt.lr},
                     {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(
                         config.opt, "process_features_lr") else config.opt.lr},
                     {'params': model.process_features_2.parameters(), 'lr': config.opt.process_features_2_lr if hasattr(
                         config.opt, "process_features_2_lr") else config.opt.lr}
                     ]
                )
        elif config.model.backbone.retrain:
            opt = torch.optim.Adam([{
                'params': model.backbone.parameters(),
                'lr': config.model.backbone.lr
            }], lr=config.opt.lr)
            opt_step = torch.optim.lr_scheduler.StepLR(opt,
                                                       step_size=8,
                                                       gamma=0.1,
                                                       verbose=True)
        else:
            opt = optim.Adam(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=config.opt.lr)

        if transformer_refine:
            opt_refine = optim.AdamW(refine_model_train.parameters(
            ), lr=config.model.transformer_refine.lr, weight_decay=0.1)

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(
        config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(
            config, type(model).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])
        if transformer_refine:
            refine_model_train = DistributedDataParallel(
                refine_model_train, device_ids=[device])
            refine_model_eval = DistributedDataParallel(
                refine_model_eval, device_ids=[device])

    if not args.eval:
        # train loop
        mpjpe_min = 100
        mpjpe_min_loop = 0
        mpjpe_re_min = 100
        mpjpe_re_min_loop = 0
        save_checkpoints = False
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            n_iters_total_train, _, _ = one_epoch(model, criterion, opt, config, train_dataloader, device, epoch, refine_model=refine_model_train, opt_refine=opt_refine,
                                                  n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer)
            if transformer_refine:
                refine_model_eval.load_state_dict(
                    refine_model_train.state_dict(), strict=False)
            n_iters_total_val, mpjpe_re, mpjpe = one_epoch(model, criterion, opt, config, val_dataloader, device, epoch, refine_model=refine_model_eval, opt_refine=opt_refine,
                                                           n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

            if master:
                checkpoint_dir = os.path.join(
                    experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_refine_dir = checkpoint_dir

                if mpjpe < mpjpe_min:
                    mpjpe_min = mpjpe
                    mpjpe_min_loop = epoch
                    save_checkpoints = True
                if mpjpe_re < mpjpe_re_min:
                    mpjpe_re_min = mpjpe_re
                    mpjpe_re_min_loop = epoch
                    save_checkpoints = True
                if save_checkpoints:
                    torch.save(model.state_dict(), os.path.join(
                        checkpoint_dir, "weights.pth"))
                    if transformer_refine:
                        torch.save(refine_model_train.state_dict(), os.path.join(
                            checkpoint_refine_dir, "refine_weights.pth"))
                    save_checkpoints = False

            print(f"{epoch} iters done. The min mpjpe is in {mpjpe_min_loop} epoch. The min mpjpe_re is in {mpjpe_re_min_loop} epoch.")

            if opt_step is not None:
                opt_step.step()
    else:
        if args.eval_dataset == 'train':
            one_epoch(model, criterion, opt, config, train_dataloader, device, 0, refine_model=refine_model_eval, opt_refine=opt_refine, n_iters_total=0,
                      is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, visible=True)
        else:
            one_epoch(model, criterion, opt, config, val_dataloader, device, 0, refine_model=refine_model_eval, opt_refine=opt_refine, n_iters_total=0,
                      is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, visible=False)


    print("Done.")


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
