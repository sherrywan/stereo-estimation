import numpy as np

import torch
from torch import nn


class Mpjpe(nn.Module):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        pjpe = torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2, dim=-1))
        loss = torch.sum(pjpe * keypoints_binary_validity.squeeze(-1))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss


class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss

class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity, if_dimension=True):
        if if_dimension:
            dimension = keypoints_pred.shape[-1]
        else: 
            dimension = len(torch.nonzero(keypoints_binary_validity))
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        if if_dimension:
            loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        else:
            loss = loss / dimension
        return loss


class KeypointsL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones((*keypoints_gt.shape[:-1], 1), device = keypoints_gt.device)
        loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2)))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss


class KeypointsL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones((*keypoints_gt.shape[:-1], 1), device = keypoints_gt.device)
        loss = torch.sum(abs(keypoints_gt - keypoints_pred))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss


class LimbDirL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.limb = [[6, 2], [6, 3], [6, 7], [2, 1], [1, 0], [3, 4], [4, 5], 
                     [7, 10], [7, 11], [7, 14], [10, 9], [9, 8], [11, 12], [12, 13]]
    
    def forward(self, keypoints_pred, limb_directions_pred):
        limb_len = len(self.limb)
        batch_num, joint_num, _ = keypoints_pred.shape
        device = keypoints_pred.device

        limbs = torch.empty((batch_num, limb_len, 3), device = device)
        for idx, limb_nodes in enumerate(self.limb):
            limbs[:,idx] = keypoints_pred[:,limb_nodes[1]] - keypoints_pred[:,limb_nodes[0]]
        limb_proj = torch.matmul(limbs.view(batch_num, limb_len, 1, 3), limb_directions_pred.view(batch_num, limb_len, 3, 1)).squeeze(-1)
        limb_proj = torch.mul(limb_proj, limb_directions_pred)

        loss = torch.mean(torch.sqrt(torch.sum((limbs-limb_proj)**2, dim=2) + 1e-13))
        return loss


class LimbL2Loss(nn.Module):
    def __init__(self, sample_num):
        super().__init__()
        self.limb = [[6, 2], [6, 3], [6, 7], [2, 1], [1, 0], [3, 4], [4, 5], 
                     [7, 10], [7, 11], [7, 14], [10, 9], [9, 8], [11, 12], [12, 13]]
        self.sample_num = sample_num
    
    def forward(self, keypoints_pred, limb_pred):
        limb_len = len(self.limb)
        batch_num, joint_num, _ = keypoints_pred.shape
        device = keypoints_pred.device

        limbs = torch.empty((batch_num, limb_len, 3), device = device)
        for idx, limb_nodes in enumerate(self.limb):
            limbs[:,idx] = keypoints_pred[:,limb_nodes[1]] - keypoints_pred[:,limb_nodes[0]]
        limbs = limbs/(self.sample_num-1)

        loss = torch.mean(torch.sqrt(torch.sum((limbs-limb_pred)**2, dim=2) + 1e-13))
        return loss
    

class SymmetryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.symmetry = [[[6,2],[6,3]],[[2,1],[3,4]],[[1,0],[4,5]],
                         [[7,10],[7,11]],[[10,9],[11,12]],[[9,8],[12,13]]]
    
    def forward(self, keypoints_pred):
        sym_len = len(self.symmetry)
        batch_num, joint_num, _ = keypoints_pred.shape
        device = keypoints_pred.device

        bone_limb = torch.zeros((batch_num, sym_len, 3), device=device)
        bone_sym = torch.zeros((batch_num, sym_len, 3), device=device)
        for idx, sym_nodes in enumerate(self.symmetry):
            bone_limb[:, idx] = keypoints_pred[:, sym_nodes[0][0]] - keypoints_pred[:, sym_nodes[0][1]]
            bone_sym[:, idx] = keypoints_pred[:, sym_nodes[1][0]] - keypoints_pred[:, sym_nodes[1][1]]
        
        loss = torch.sum(abs(torch.sqrt(torch.sum(bone_limb**2,dim=2)) - torch.sqrt(torch.sum(bone_sym**2,dim=2))))
        loss = loss / (batch_num * sym_len)
        return loss


class DepthSmoothLoss(nn.Module):
    def __init__(self, threshold_b=500, threshold_s=1500):
        super().__init__()
        self.threshold_b = threshold_b
        self.threshold_s = threshold_s

    def forward(self, keypoints_pred, keypoints_depth, keypoints_binary_validity=None):
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones((*keypoints_depth.shape[:-1], 1), device = keypoints_depth.device)

        diff = torch.sqrt(torch.sum((keypoints_depth - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2))
        diff_c = torch.empty_like(diff)
        diff_c = diff.clone()
        diff_c[diff < self.threshold_b] = torch.pow(diff[diff < self.threshold_b], 0.1)
        diff_c[diff > self.threshold_s] = torch.pow(diff[diff > self.threshold_s], 0.1) * (self.threshold_s ** 0.9)
        
        keypoints_binary_validity_c = keypoints_binary_validity.clone()
        keypoints_binary_validity_c[diff < self.threshold_b] = 0

        loss = torch.sum(diff_c) / (max(1, torch.sum(keypoints_binary_validity_c).item()))
        
        return loss


class VolumetricCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity, model='vol'):
        loss = 0.0
        n_losses = 0

        batch_size = volumes_batch_pred.shape[0]
        for batch_i in range(batch_size):
            if model == 'vol':
                coord_volume = coord_volumes_batch[batch_i]
            elif model == 'stereo':
                coord_volume = coord_volumes_batch
            keypoints_gt_i = keypoints_gt[batch_i]

            coord_volume_unsq = coord_volume.unsqueeze(0)
            keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            dists = dists.view(dists.shape[0], -1)

            min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
            min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)

            for joint_i, index in enumerate(min_indexes):
                validity = keypoints_binary_validity[batch_i, joint_i]
                # only focus on center point
                # loss += validity[0] * (-torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-6))
                # focus on the 9-neighbor of center point and others points 
                focus_area = volumes_batch_pred[batch_i, joint_i, max(index[0]-1, 0):min(index[0]+2, 16), max(index[1]-1, 0):min(index[1]+2, 64), max(index[2]-1, 0): min(index[2]+2, 64)]
                loss += validity[0] * (- torch.log(torch.sum(focus_area) + 1e-6)
                                       + torch.log(torch.sum(volumes_batch_pred) - torch.sum(focus_area)))
                n_losses += 1


        return loss / n_losses


class HeatmapCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_gt, heatmap):
        loss = 0.0
        n_losses = 0

        batch_size = keypoints_gt.shape[0]
        view_num = keypoints_gt.shape[1]
        joint_num = keypoints_gt.shape[2]
        heatmap_shape = heatmap.shape[3]
        for batch_i in range(batch_size):       
            for view_i in range(view_num):    
                keypoints_gt_i = keypoints_gt[batch_i, view_i].ceil()
                for joint_i in range(joint_num):       
                    # only focus on center point
                    loss += (-torch.log(heatmap[batch_i, view_i, 0, min(max(int(keypoints_gt_i[joint_i, 1].item()), 0), heatmap_shape-1), min(max(int(keypoints_gt_i[joint_i, 0].item()), 0), heatmap_shape-1)]))
                    n_losses += 1

        return loss / n_losses