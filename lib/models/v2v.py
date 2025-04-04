# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch

import torch.nn as nn
import torch.nn.functional as F


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

        if layers >= 3:
            self.encoder_pool3 = Pool3DBlock(2)
            self.encoder_res3 = Res3DBlock(128, 128)
            self.decoder_res3 = Res3DBlock(128, 128)
            self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2)
            self.skip_res3 = Res3DBlock(128, 128)
        if layers >= 4:
            self.encoder_pool4 = Pool3DBlock(2)
            self.encoder_res4 = Res3DBlock(128, 128)
            self.decoder_res4 = Res3DBlock(128, 128)
            self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2)
            self.skip_res4 = Res3DBlock(128, 128)
        if layers >= 5:
            self.encoder_pool5 = Pool3DBlock(2)
            self.encoder_res5 = Res3DBlock(128, 128)
            self.decoder_res5 = Res3DBlock(128, 128)
            self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2)
            self.skip_res5 = Res3DBlock(128, 128)  
        
    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        if self.layers>=3:
            skip_x3 = self.skip_res3(x)
            x = self.encoder_pool3(x)
            x = self.encoder_res3(x)
        if self.layers>=4:
            skip_x4 = self.skip_res4(x)
            x = self.encoder_pool4(x)
            x = self.encoder_res4(x)
        if self.layers>=5:
            skip_x5 = self.skip_res5(x)
            x = self.encoder_pool5(x)
            x = self.encoder_res5(x)

        x = self.mid_res(x)

        if self.layers>=5:
            x = self.decoder_res5(x)
            x = self.decoder_upsample5(x)
            x = x + skip_x5
        if self.layers>=4:
            x = self.decoder_res4(x)
            x = self.decoder_upsample4(x)
            x = x + skip_x4
        if self.layers>=3:
            x = self.decoder_res3(x)
            x = self.decoder_upsample3(x)
            x = x + skip_x3

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2),
            nn.ReLU(inplace=True)

        )

        self.head = nn.Sequential(
            nn.Conv1d(16, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, J, C, D, H, W = x.shape
        x = x.view(-1, C, D, H, W)
        x = self.features(x)

        C = x.shape[1]
        x = x.view((B, J, C, -1))
        x = x.mean(dim=-1)
        x = x.permute(0,2,1) #(b,c,j)

        out = self.head(x)
        out = out.permute(0,2,1) #(b,j,1)

        return out
    

class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels, out_sep=False, layers=5, if_conf=False):
        super().__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )

        self.encoder_decoder = EncoderDecorder(layers)

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )
        self.out_sep = out_sep
        if not out_sep:
            self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self.if_conf = if_conf
        if if_conf:
            self.confidences = GlobalAveragePoolingHead(32)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        
        if self.if_conf:
            feas = x

        x = self.back_layers(x)
        if not self.out_sep:
            x = self.output_layer(x)
        
        confidences = None
        if self.if_conf:
            hm = x
            hm = hm.unsqueeze(2)
            feas = feas.unsqueeze(1)
            feas = feas * hm  #(b,j,c,d,h,w)
            confidences = self.confidences(feas) #(b,j,1)

        return x, confidences

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class VHModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.output_layer = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)