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
import random

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
from lib.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss, Mpjpe
from lib.models.mask_transofrmer import Masked_ST
from lib.utils import img, multiview, op, vis, misc, cfg
from lib.datasets import mhad_stereo_for_MaskST, human36m_for_MaskST
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


def setup_mhad_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = mhad_stereo_for_MaskST.MHADStereoViewDataset(
            mhad_root=config.dataset.train.mhad_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(
                config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(
                config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            scale_bbox=config.dataset.train.scale_bbox,
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
            collate_fn=dataset_utils.make_collate_fn_for_MaskST(randomize_n_views=config.dataset.train.randomize_n_views,
                                                                min_n_views=config.dataset.train.min_n_views,
                                                                max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=False
        )

    # val
    val_dataset = mhad_stereo_for_MaskST.MHADStereoViewDataset(
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
        collate_fn=dataset_utils.make_collate_fn_for_MaskST(randomize_n_views=config.dataset.val.randomize_n_views,
                                                            min_n_views=config.dataset.val.min_n_views,
                                                            max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=False
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = human36m_for_MaskST.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(
                config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(
                config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.h36m_labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            scale_bbox=config.dataset.train.scale_bbox,
            kind="human36m",
            undistort_images=True,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(
                config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(
                config.dataset.train, "crop") else True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (
                train_sampler is None),  # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn_for_MaskST(randomize_n_views=config.dataset.train.randomize_n_views,
                                                                min_n_views=config.dataset.train.min_n_views,
                                                                max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = human36m_for_MaskST.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(
            config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(
            config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.h36m_labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind="human36m",
        undistort_images=True,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(
            config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(
            config.dataset.val, "crop") else True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(
            config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn_for_MaskST(randomize_n_views=config.dataset.val.randomize_n_views,
                                                            min_n_views=config.dataset.val.min_n_views,
                                                            max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, dataset_type='mhad', is_train=True, distributed_train=False):
    if dataset_type == 'mhad':
        train_dataloader, val_dataloader, train_sampler = setup_mhad_dataloaders(
            config, is_train, distributed_train)
    elif dataset_type == 'h36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(
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


def generate_binary_sequence(mask, p):
    num = mask.shape[1]
    for i in range(num):
        rand_num = random.random()  # 生成介于 0 和 1 之间的随机数
        mask[:, i] = 1 if rand_num <= p else 0  # 根据概率生成元素
    return mask


def one_epoch(criterion, config, dataloader, device, epoch, refine_model=None, mask_ratio=0, opt_refine=None, n_iters_total=0, is_train=True, caption='', master=False, experiment_dir=None, writer=None, visible=False, dataset_type='mhad', K=1):
    name = "train" if is_train else "val"
    model_type = config.model.name

    if is_train:
        refine_model.train()
    else:
        refine_model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        iterator = enumerate(dataloader)
        # if is_train and config.opt.n_iters_per_epoch is not None:
        #     iterator = islice(iterator, config.opt.n_iters_per_epoch)

        for iter_i, batch in iterator:
            with autograd.detect_anomaly():
                # measure data loading time
                data_time = time.time() - end

                if batch is None:
                    print("Found None batch")
                    continue

                keypoints_3d_gt, keypoints_3d_validity_gt = dataset_utils.prepare_batch_for_pretrain(
                    batch, device, config)

                if dataset_type == "h36m":
                    keypoints_3d_gt = keypoints_3d_gt[:, :, [0, 2, 1]]

                batch_size = keypoints_3d_gt.shape[0]
                n_joints = keypoints_3d_gt.shape[1]

                if config.model.transformer_refine.relative_pose:
                    keypoints_root = keypoints_3d_gt[:, 6:7, :].clone()
                    keypoints_3d_gt = keypoints_3d_gt - keypoints_root

                keypoints_3d_binary_validity_gt = (
                    keypoints_3d_validity_gt > 0.0).type(torch.float32)

                # generate mask, 1 for masked, 0 for unmasked
                keypoints_3d_mask = torch.ones_like(
                    keypoints_3d_binary_validity_gt)
                keypoints_3d_mask = generate_binary_sequence(
                    keypoints_3d_mask, mask_ratio)

                masked_joints = torch.tensor(
                    [0, 10000, 0]).float().cuda().to(device)
                keypoints_3d_pred = torch.where(
                    keypoints_3d_mask == 1, masked_joints, keypoints_3d_gt)
                
                keypoints_3d_pred_refined_list = []
                if refine_model is not None:
                    keypoints_3d_pred_refined, attention = refine_model(
                        keypoints_3d_pred, None)
                    keypoints_3d_pred_refined_list.append(keypoints_3d_pred_refined)
                    for repeat_k in range(K):
                        with torch.no_grad():
                            joints_confidence = torch.sum(attention[-1], dim=(1, 2)).reshape(batch_size, -1, 1)
                            masked_conf = torch.zeros_like(joints_confidence)
                            masked_confidence = torch.where(
                                keypoints_3d_mask == 1, joints_confidence, masked_conf)
                            _, topk_index = torch.topk(
                                masked_confidence, 2, dim=1)
                            for batch_i in range(batch_size):
                                keypoints_3d_mask[batch_i, topk_index[batch_i]] = 0
                        keypoints_3d_pred_1 = torch.where(
                            keypoints_3d_mask == 1, masked_joints, keypoints_3d_pred_refined)
                        keypoints_3d_pred_refined, attention = refine_model(
                            keypoints_3d_pred_1, None)
                        keypoints_3d_pred_refined_list.append(keypoints_3d_pred_refined)

                # calculate loss
                total_loss = 0.0
                if refine_model is not None:
                    for keypoints_3d_pred_refined in keypoints_3d_pred_refined_list:
                        loss = criterion(keypoints_3d_pred_refined,
                                        keypoints_3d_gt, keypoints_3d_binary_validity_gt)
                        total_loss += loss
                # if refine_model is not None:
                #     loss = criterion(keypoints_3d_pred_refined,
                #                      keypoints_3d_gt, keypoints_3d_binary_validity_gt)
                # total_loss += loss

                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                metric_dict['total_loss'].append(total_loss.item())

                if is_train:
                    if refine_model is not None and opt_refine is not None:
                        opt_refine.zero_grad()

                    total_loss.backward()

                    if refine_model is not None and opt_refine is not None:
                        opt_refine.step()

                if iter_i % 10 == 0:
                    print("iter{}: Loss:{}, ".format(
                        iter_i, total_loss.item()))

                if config.model.transformer_refine.relative_pose:
                    keypoints_3d_pred_refined = keypoints_3d_pred_refined + keypoints_root
                # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(
                        keypoints_3d_pred_refined.detach().cpu().numpy())
                    results['keypoints_3d_gt'].append(
                        keypoints_3d_gt.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])

                # plot visualization
                if master and visible:
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
                                      2, n_iters_total)

                    n_iters_total += 1

    # calculate evaluation metrics
    scalar_metric = None
    if master:
        if not is_train:
            results['keypoints_3d'] = np.concatenate(
                results['keypoints_3d'], axis=0)
            np.save(os.path.join(experiment_dir, "keypoints_3d_pred.npy"),
                    results['keypoints_3d'])
            results['keypoints_3d_gt'] = np.concatenate(
                results['keypoints_3d_gt'], axis=0)
            np.save(os.path.join(experiment_dir, "keypoints_3d_gt.npy"),
                    results['keypoints_3d_gt'])
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric, full_metric = dataloader.dataset.evaluate(
                    results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)

            checkpoint_dir = os.path.join(
                experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            # dump full metric
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    return n_iters_total, scalar_metric


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

    mask_iter_Ks = [1]

    for mask_iter_K in mask_iter_Ks:
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

        # refine_transformer
        transformer_refine = config.model.if_transofrmer_refine

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
        opt_refine = None
        opt_exp = None
        if not args.eval:
            if transformer_refine:
                opt_refine = optim.AdamW(refine_model_train.parameters(
                ), lr=config.model.transformer_refine.lr, weight_decay=0.1)
                # opt_exp = torch.optim.lr_scheduler.ExponentialLR(opt_refine, gamma=0.98)

        # datasets
        print("Loading data...")
        train_dataloader_mhad, val_dataloader_mhad, train_sampler_mhad = setup_dataloaders(
            config, dataset_type='mhad', distributed_train=is_distributed)
        train_dataloader_h36m, val_dataloader_h36m, train_sampler_h36m = setup_dataloaders(
            config, dataset_type='h36m', distributed_train=is_distributed)

        # experiment
        experiment_dir, writer = None, None
        if master:
            experiment_dir, writer = setup_experiment(
                config, "MaskST_{}".format(mask_iter_K), is_train=not args.eval)

        # multi-gpu
        if is_distributed:
            refine_model_train = DistributedDataParallel(
                refine_model_train, device_ids=[device])
            refine_model_eval = DistributedDataParallel(
                refine_model_eval, device_ids=[device])

        if not args.eval:
            # train loop
            n_iters_total_train, n_iters_total_val = 0, 0
            mpjpe_min = 50
            epoch_min = 0
            for epoch in range(config.opt.n_epochs):
                p_list = [0.1, 0.2, 0.3, 0.4]
                p = p_list[epoch % 4]
                p_eval = 0.1
                if train_sampler_mhad is not None:
                    train_sampler_mhad.set_epoch(epoch)

                if train_sampler_h36m is not None:
                    train_sampler_mhad.set_epoch(epoch)
                print("Start training in h36m!")
                n_iters_total_train, _ = one_epoch(criterion, config, train_dataloader_h36m, device, epoch, refine_model=refine_model_train, opt_refine=opt_refine, mask_ratio=p,
                                                n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer, dataset_type="h36m", K=mask_iter_K)
                print("Start evaluating in h36m!")
                refine_model_eval.load_state_dict(
                    refine_model_train.state_dict(), strict=False)
                n_iters_total_val, mpjpe = one_epoch(criterion, config, val_dataloader_h36m, device, epoch, refine_model=refine_model_eval, opt_refine=opt_refine, mask_ratio=p_eval,
                                                n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, dataset_type="h36m", K=mask_iter_K)

                print("Start training in mhad!")
                n_iters_total_train, _ = one_epoch(criterion, config, train_dataloader_mhad, device, epoch, refine_model=refine_model_train, opt_refine=opt_refine, mask_ratio=p,
                                                n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer, K=mask_iter_K)
                print("Start evaluating in mhad!")
                refine_model_eval.load_state_dict(
                    refine_model_train.state_dict(), strict=False)
                n_iters_total_val, _ = one_epoch(criterion, config, val_dataloader_mhad, device, epoch, refine_model=refine_model_eval, opt_refine=opt_refine, mask_ratio=p_eval,
                                                    n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, K=mask_iter_K)

                if master:
                    checkpoint_refine_dir = os.path.join(
                        experiment_dir, "checkpoints", "{:04}".format(epoch))
                    os.makedirs(checkpoint_refine_dir, exist_ok=True)
                    if mpjpe < mpjpe_min:
                        mpjpe_min = mpjpe
                        epoch_min = epoch
                        torch.save(refine_model_train.state_dict(), os.path.join(
                            checkpoint_refine_dir, "refine_weights.pth"))

                print(f"{epoch} iters done.")
                print(
                    f"The best results is in {epoch_min}th epoch with {mpjpe_min} mpjpe!")

                if opt_exp is not None and (epoch % 4 == 0):
                    opt_exp.step()
        else:
            if args.eval_dataset == 'train':
                one_epoch(criterion, config, train_dataloader, device, 0, refine_model=refine_model_eval, n_iters_total=0,
                        is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)
            else:
                one_epoch(criterion, config, val_dataloader, device, 0, refine_model=refine_model_eval, n_iters_total=0,
                        is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, visible=False)

    print("Done.")


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
