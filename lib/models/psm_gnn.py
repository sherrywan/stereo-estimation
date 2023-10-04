import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F



class GlobalAveragePoolingHead_3d(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)

        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, -1))
        x = x.mean(dim=-1)

        out = self.head(x)

        return out


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size,
                               stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()

        self.encoder_res1 = Res3DBlock(in_planes, 32)
        self.decoder_res1 = Res3DBlock(32, 32)
        self.decoder_upsample1 = Upsample3DBlock(32, out_planes, 2, 2)

        self.encoder_res2 = Res3DBlock(32, 64)
        self.decoder_res2 = Res3DBlock(64, 64)
        self.decoder_upsample2 = Upsample3DBlock(64, 32, 2, 2)

        # self.skip_res1 = Res3DBlock(in_planes, out_planes)
        # self.skip_res2 = Res3DBlock(32, 32)

    def forward(self, x):
        # skip_x1 = self.skip_res1(x)
        x = self.encoder_res1(x)
        # skip_x2 = self.skip_res2(x)
        x = self.encoder_res2(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        # x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        # x = x + skip_x1

        return x


class PSMGNNModel(nn.Module):
    def __init__(self, input_channels, output_channels, json_data, data_dir, device, config_psm):
        super().__init__()

        vol_confidence=config_psm.vol_confidence
        self.temperature=config_psm.temperature
        self.prob_softmax=config_psm.prob_softmax
        self.skeleton = json_data
        self.if_conv = config_psm.conv_tag
        self.conv_init = config_psm.conv_init

        self.asc_skeleton_sorted_by_level, self.desc_skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)
        
        if vol_confidence:
            self.vol_confidence = GlobalAveragePoolingHead_3d(
                input_channels, input_channels)

        self.encoder_res1 = Res3DBlock(input_channels, 32)
        self.decoder_res1 = Res3DBlock(32, 32)
        self.decoder_upsample1 = Upsample3DBlock(32, output_channels, 2, 2)

        self.encoder_res2 = Res3DBlock(32, 64)
        self.decoder_res2 = Res3DBlock(64, 64)
        self.decoder_upsample2 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(input_channels, output_channels)
        self.skip_res2 = Res3DBlock(32, 32)
 
        if self.if_conv:
            self.encoder_pool1 = Pool3DBlock(2)
            self.encoder_pool2 = Pool3DBlock(2)
            
        self._initialize_weights()

        self.pairwise_attention_net = []
        for data in self.skeleton:
            name = data['name']
            childs = data['children']
            data['pairwise'] = []
            for child in childs:
                if self.if_conv:
                    pairwise_attention = None
                    if self.conv_init:
                        pairwise_attention = np.load(os.path.join(data_dir, f"{name}_{child}_pairattention_conv_temp{self.temperature}.npy"))
                        pairwise_attention = torch.from_numpy(pairwise_attention).float()
                    self.pairwise_attention_net.append(self.make_conv_with_init(33, pairwise_attention.unsqueeze(0).unsqueeze(0)))
                    data['pairwise'].append(self.pairwise_attention_net[-1])
                else:
                    pairwise_attention = np.load(os.path.join(data_dir, f"{name}_{child}_pairattention_temp{self.temperature}.npy"))
                    pairwise_attention = torch.from_numpy(pairwise_attention).float().cuda(device)
                    data['pairwise'].append(pairwise_attention)   
        if self.if_conv:
            self.pairwise_attention_net = nn.Sequential(*self.pairwise_attention_net)    


    def make_conv_with_init(self, kernel, initial_data=None):
        conv = nn.Conv3d(1, 1, kernel_size = kernel, stride=1, padding=16)
        if initial_data is not None:
            conv.weight.data = initial_data
        return conv


    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[6]]
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]
        
        asc_order = np.argsort(level)
        asc_sorted_skeleton = []
        for i in asc_order:
            skeleton[i]['level'] = level[i]
            asc_sorted_skeleton.append(skeleton[i])

        desc_order = np.argsort(level)[::-1]
        desc_sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            desc_sorted_skeleton.append(skeleton[i])
        return asc_sorted_skeleton, desc_sorted_skeleton

    def forward(self, x, volume_3d_multiplier = 1):
        batch_size, joints_num, volume_size_x, volume_size_y, volume_size_z = x.shape

        leaf_to_root_skeleton = self.desc_skeleton_sorted_by_level
        root_to_leaf_skeleton = self.asc_skeleton_sorted_by_level

        if self.if_conv:
            # probability forward
            x1 = x.clone()
            for node in leaf_to_root_skeleton:
                node_idx = node['idx']
                node_prob = x[:, node_idx:(node_idx+1)]
                if len(node['children']) == 0:
                    continue
                else:
                    for child_i, child in enumerate(node['children']):
                        child_prob = x[:, child:(child+1)]
                        max_v = node['pairwise'][child_i](child_prob)
                        node_prob = node_prob * max_v
                    if self.prob_softmax:
                        node_prob = F.softmax(node_prob.reshape(batch_size, -1), dim=1)
                    x1[:,node_idx] = node_prob.reshape(batch_size, volume_size_x, volume_size_y, volume_size_z)
            
            # probability backward
            x2 = x1.clone()
            for node in root_to_leaf_skeleton:
                node_idx = node['idx']
                node_prob = x1[:, node_idx:(node_idx+1)]
                if len(node['children']) == 0:
                    continue
                else:
                    for child_i, child in enumerate(node['children']):
                        child_prob = x1[:, child:(child+1)]
                        max_v = node['pairwise'][child_i](node_prob)
                        child_prob = child_prob * max_v
                        if self.prob_softmax:
                            child_prob = F.softmax(child_prob.reshape(batch_size, -1), dim=1)
                        x2[:, child] = child_prob.reshape(batch_size, volume_size_x, volume_size_y, volume_size_z)
            
            skip_x1 = self.skip_res1(x2)
            x2 = self.encoder_pool1(x2)
            x2 = self.encoder_res1(x2)

            skip_x2 = self.skip_res2(x2)
            x2 = self.encoder_pool2(x2)
            x2 = self.encoder_res2(x2)

            x2 = self.decoder_res2(x2)
            x2 = self.decoder_upsample2(x2)
            x2 = x2 + skip_x2

            x2 = self.decoder_res1(x2)
            x2 = self.decoder_upsample1(x2)
            x2 = x2 + skip_x1

        else:
            skip_x1 = self.skip_res1(x)
            x = F.interpolate(x , (int(volume_size_x/2), int(volume_size_y/2), int(volume_size_z/2)), mode='trilinear')
            x = F.softmax(x.reshape(batch_size, joints_num, -1) * volume_3d_multiplier, dim=2)
            x = x.reshape(batch_size, joints_num, int(volume_size_x/2), int(volume_size_y/2), int(volume_size_z/2))
            
            skip_x2 = self.skip_res2(x)
            x = F.interpolate(x , (int(volume_size_x/4), int(volume_size_y/4), int(volume_size_z/4)), mode='trilinear')
            x = F.softmax(x.reshape(batch_size, joints_num, -1) * volume_3d_multiplier, dim=2)
            x = x.reshape(batch_size, joints_num, int(volume_size_x/4), int(volume_size_y/4), int(volume_size_z/4))
        
            if hasattr(self, "vol_confidences"):
                vol_confidences = self.vol_confidence(x)
                x = x * vol_confidences
        
            # probability forward
            x1 = x.clone()
            for node in leaf_to_root_skeleton:
                node_idx = node['idx']
                node_prob = x[:, node_idx]
                node_prob = node_prob.reshape(batch_size, -1)
                if len(node['children']) == 0:
                    continue
                else:
                    for child_i, child in enumerate(node['children']):
                        child_prob = x[:, child]
                        child_prob = child_prob.reshape(batch_size, 1, -1)
                        pw = node['pairwise'][child_i].unsqueeze(0)
                        pwcp = child_prob * pw
                        max_v = torch.sum(pwcp, dim=2)
                        # node_prob_new = node_prob.clone()
                        node_prob = node_prob * max_v.squeeze(0)
                        # node_prob = node_prob_new
                    if self.prob_softmax:
                        node_prob = F.softmax(node_prob, dim=1)
                    x1[:,node_idx] = node_prob.reshape(batch_size, int(volume_size_x/4), int(volume_size_y/4), int(volume_size_z/4))
            
            # probability backward
            x2 = x1.clone()
            for node in root_to_leaf_skeleton:
                node_idx = node['idx']
                node_prob = x1[:, node_idx]
                node_prob = node_prob.reshape(batch_size, 1, -1)
                if len(node['children']) == 0:
                    continue
                else:
                    for child_i, child in enumerate(node['children']):
                        child_prob = x1[:, child]
                        child_prob = child_prob.reshape(batch_size, -1)
                        pw = node['pairwise'][child_i].unsqueeze(0)
                        pwrp = node_prob * pw
                        max_v = torch.sum(pwrp, dim=2)
                        # child_prob_new = child_prob.clone()
                        child_prob = child_prob * max_v.squeeze(0)
                        # child_prob = child_prob_new
                        if self.prob_softmax:
                            child_prob = F.softmax(child_prob, dim=1)
                        x2[:, child] = child_prob.reshape(batch_size, int(volume_size_x/4), int(volume_size_y/4), int(volume_size_z/4))

            x2 = self.encoder_res1(x2)
            x2 = self.encoder_res2(x2)

            x2 = self.decoder_res2(x2)
            x2 = self.decoder_upsample2(x2)
            x2 = x2 + skip_x2

            x2 = self.decoder_res1(x2)
            x2 = self.decoder_upsample1(x2)
            x2 = x2 + skip_x1

        return x2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class PSMGNNModel_nolearn(nn.Module):
    def __init__(self, input_channels, output_channels, json_data, data_dir, device, config_psm):
        super().__init__()

        vol_confidence=config_psm.vol_confidence
        self.temperature=config_psm.temperature
        self.prob_softmax=config_psm.prob_softmax
        self.skeleton = json_data
        self.if_conv = config_psm.conv_tag
        self.conv_init = config_psm.conv_init

        self.asc_skeleton_sorted_by_level, self.desc_skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)
        
        self.pairwise_attention_net = []
        for data in self.skeleton:
            name = data['name']
            childs = data['children']
            data['pairwise'] = []
            for child in childs:
                pairwise_attention = None
                if self.conv_init:
                    pairwise_attention = np.load(os.path.join(data_dir, f"{name}_{child}_pairattention_conv_temp{self.temperature}_wosoftmax.npy"))
                    pairwise_attention = torch.from_numpy(pairwise_attention).float()
                self.pairwise_attention_net.append(self.make_conv_with_init(33, pairwise_attention.unsqueeze(0).unsqueeze(0)))
                data['pairwise'].append(self.pairwise_attention_net[-1])

        self.pairwise_attention_net = nn.Sequential(*self.pairwise_attention_net)    


    def make_conv_with_init(self, kernel, initial_data=None):
        conv = nn.Conv3d(1, 1, kernel_size = kernel, stride=1, padding=16)
        if initial_data is not None:
            conv.weight.data = initial_data
        return conv


    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[6]]
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]
        
        asc_order = np.argsort(level)
        asc_sorted_skeleton = []
        for i in asc_order:
            skeleton[i]['level'] = level[i]
            asc_sorted_skeleton.append(skeleton[i])

        desc_order = np.argsort(level)[::-1]
        desc_sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            desc_sorted_skeleton.append(skeleton[i])
        return asc_sorted_skeleton, desc_sorted_skeleton

    def forward(self, x, occlusion, volume_3d_multiplier = 1):
        batch_size, joints_num, volume_size_x, volume_size_y, volume_size_z = x.shape

        leaf_to_root_skeleton = self.desc_skeleton_sorted_by_level
        root_to_leaf_skeleton = self.asc_skeleton_sorted_by_level

        occluded = np.nonzero(occlusion==1)
        # probability forward
        x1 = x.clone()
        if len(occluded[0]) > 0:
            x1[occluded[0], occluded[1]] = 0.5 * x1[occluded[0], occluded[1]]
        for node in leaf_to_root_skeleton:
            node_idx = node['idx']
            node_prob = x[:, node_idx:(node_idx+1)]
            if len(node['children']) == 0:
                continue
            else:
                for child_i, child in enumerate(node['children']):
                    child_prob = x[:, child:(child+1)]
                    max_v = node['pairwise'][child_i](child_prob)
                    node_prob = node_prob * max_v
                # if self.prob_softmax:
                #     node_prob = F.softmax(node_prob.reshape(batch_size, -1), dim=1)
                x1[:,node_idx] = node_prob.reshape(batch_size, volume_size_x, volume_size_y, volume_size_z)
        
        # probability backward
        x2 = x1.clone()
        for node in root_to_leaf_skeleton:
            node_idx = node['idx']
            node_prob = x1[:, node_idx:(node_idx+1)]
            if len(node['children']) == 0:
                continue
            else:
                for child_i, child in enumerate(node['children']):
                    child_prob = x1[:, child:(child+1)]
                    max_v = node['pairwise'][child_i](node_prob)
                    child_prob = child_prob * max_v
                    # if self.prob_softmax:
                    #     child_prob = F.softmax(child_prob.reshape(batch_size, -1), dim=1)
                    x2[:, child] = child_prob.reshape(batch_size, volume_size_x, volume_size_y, volume_size_z)

        return x2

