from typing_extensions import Required

import train
from lib.utils import  vis, cfg
from lib.datasets import utils as dataset_utils
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument("--index", type=int, required=True, help="Index, the idx of pose")
    parser.add_argument("--res_folder", type=str, required=True, help="Path, where keypoints results are stored")
    parser.add_argument('--method', type=str, default='ltri', help="Method used")
    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--logdir", type=str, default="./logs", help="Path, where logs will be stored")
    parser.add_argument("--vis_type", type=str, default="val", help="dataset kinds")

    args = parser.parse_args()
    return args

def vis_all(data, keypoints_2d, keypoints_3d, config, vis_folder = './res/vis', idx=0):
    os.makedirs(vis_folder, exist_ok=True)
    # draw picture
    fig = plt.figure(figsize=(12,12), dpi=1200)
    axes = fig.subplots(nrows=1, ncols=2)
    plt.axis('off')
    axes[0].imshow(data[0][:,:,::-1])
    axes[1].imshow(data[1][:,:,::-1])
    plt.savefig(os.path.join(vis_folder, "binocular_images_{}.png".format(idx)))
    plt.close()
    # draw 3D pose
    keypoints_3d_re = keypoints_3d[:,:] - keypoints_3d[6:7, :]
    fig = plt.figure(figsize=(16,6), dpi=1200)
    axes = fig.subplots(nrows=1, ncols=4, subplot_kw=dict(fc='whitesmoke', projection='3d',))
    vis.draw_3d_pose_in_two_views(axes, keypoints_3d_pred=keypoints_3d_re)
    plt.savefig(os.path.join(vis_folder, "keypoints_3d_{}_{}.png".format("gt", idx)))
    plt.close()
    print("Successfully draw 3d result of {}th pose.".format(idx))
    

def main(args_path):     
    # config
    config = cfg.load_config(args_path)
    config.opt.batch_size = 1
    config.dataset.norm=False
    is_distributed = False

    # datasets
    print("Loading data...")
    train_dataloader, _, train_sampler = train.setup_dataloaders(config, distributed_train=is_distributed)

    for iter_i, batch in enumerate(train_dataloader):
        if iter_i % 2000:
            idxs = batch['indexes']
            images = batch['images']
            keypoints_2d = batch['keypoints_2d']
            keypoints_3d = batch['keypoints_3d']
    
            vis_all(images[0], keypoints_2d[0], keypoints_3d[0], config, vis_folder="./vis/gt", idx=idxs[0])


if __name__ == '__main__':
    args_path = '/data0/wxy/3d_pose/stereo-estimation/experiments/mhad/train/mhad_stereo_volume_resnet152.yaml'
    main(args_path)
