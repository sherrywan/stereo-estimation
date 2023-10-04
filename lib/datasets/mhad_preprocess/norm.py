'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-20 23:07:27
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-03-21 11:18:56
FilePath: /wxy/3d_pose/stereo-estimation/lib/datasets/mhad_preprocess/norm.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import torch
from torchvision.datasets import ImageFolder
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from lib.datasets.mhad_stereo import MHADStereoViewDataset


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_batches = 0
    for data in train_loader:
        mean += torch.mean(data.float().div(255),dim=[0,1,2,3])
        std += torch.mean(data.float().div(255)**2,dim=[0,1,2,3])
        num_batches+=1

    e_x = mean/num_batches
    #计算E(X^2)
    e_x_squared=std/num_batches
    #计算var(X)=E(X^2)]-[E(X)]^2
    var=e_x_squared-e_x**2
    print(e_x, var**0.5)


if __name__ == '__main__':
    mhad_root = '/data1/share/dataset/MHAD_Berkeley/stereo_camera/'
    labels_stereo_npy_path = '/data1/share/dataset/MHAD_Berkeley/stereo_camera/extra/human36m-stereo-m-labels-GTbboxes.npy'
    
    dataset = MHADStereoViewDataset(
        mhad_root,
        labels_stereo_npy_path,
        train=True,                       # include all possible data
        test=True,
        image_shape=None,                 # don't resize
        retain_every_n_frames_in_test=1,  # yes actually ALL possible data
        kind="mhad",
        scale_bbox=1.0,
        norm_image=False,
        crop=False,        
        rectificated=False,
        baseline_width='m')  
    print(getStat(dataset))
