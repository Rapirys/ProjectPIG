import torch
from torch import nn
from typing import Tuple

#source: https://github.com/astra-vision/MonoScene/blob/master/monoscene/models/DDR.py
"""
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
"""
class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        bn_momentum=0.0003,
    ):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False
        )
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu

class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[
                Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(
            feature,
            feature // 4,
            bn_momentum=bn_momentum,
            expansion=expansion,
            stride=2,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    feature,
                    int(feature * expansion / 4),
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                norm_layer(int(feature * expansion / 4), momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.main(x)

class ASPP(nn.Module):
    """
    ASPP 3D
    Adapt from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, planes, dilations_conv_list):
        super().__init__()

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_in):

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        return x_in


#Source: https://github.com/astra-vision/MonoScene/blob/master/monoscene/models/CRP3D.py
class CPMegaVoxels(nn.Module):
    def __init__(self, feature, size, n_relations=4, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_relations = n_relations
        self.flatten_size = size[0] * size[1] * size[2]
        self.feature = feature
        self.context_feature = feature * 2
        self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)
        padding = ((size[0] + 1) % 2, (size[1] + 1) % 2, (size[2] + 1) % 2)

        self.mega_context = nn.Sequential(
            nn.Conv3d(
                feature, self.context_feature, stride=2, padding=padding, kernel_size=3
            ),
        )
        self.flatten_context_size = (size[0] // 2) * (size[1] // 2) * (size[2] // 2)

        self.context_prior_logits = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        self.feature,
                        self.flatten_context_size,
                        padding=0,
                        kernel_size=1,
                    ),
                )
                for i in range(n_relations)
            ]
        )
        self.aspp = ASPP(feature, [1, 2, 3])

        self.resize = nn.Sequential(
            nn.Conv3d(
                self.context_feature * self.n_relations + feature,
                feature,
                kernel_size=1,
                padding=0,
                bias=False,
                ),
            Process(feature, nn.BatchNorm3d, bn_momentum, dilations=[1]),
        )

    def forward(self, input):
        ret = {}
        bs = input.shape[0]

        x_agg = self.aspp(input)

        # get the mega context
        x_mega_context_raw = self.mega_context(x_agg)
        x_mega_context = x_mega_context_raw.reshape(bs, self.context_feature, -1)
        x_mega_context = x_mega_context.permute(0, 2, 1)

        # get context prior map
        x_context_prior_logits = []
        x_context_rels = []
        for rel in range(self.n_relations):

            # Compute the relation matrices
            x_context_prior_logit = self.context_prior_logits[rel](x_agg)
            x_context_prior_logit = x_context_prior_logit.reshape(
                bs, self.flatten_context_size, self.flatten_size
            )
            x_context_prior_logits.append(x_context_prior_logit.unsqueeze(1))

            x_context_prior_logit = x_context_prior_logit.permute(0, 2, 1)
            x_context_prior = torch.sigmoid(x_context_prior_logit)

            # Multiply the relation matrices with the mega context to gather context features
            x_context_rel = torch.bmm(x_context_prior, x_mega_context)  # bs, N, f
            x_context_rels.append(x_context_rel)

        x_context = torch.cat(x_context_rels, dim=2)
        x_context = x_context.permute(0, 2, 1)
        x_context = x_context.reshape(
            bs, x_context.shape[1], self.size[0], self.size[1], self.size[2]
        )

        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)

        x_context_prior_logits = torch.cat(x_context_prior_logits, dim=1)
        ret["P_logits"] = x_context_prior_logits
        ret["x"] = x

        return ret


## Source: https://github.com/astra-vision/MonoScene/blob/master/monoscene/models/flosp.py
class FLoSP(nn.Module):

    #Input:
    # x2d: 2D feature map of shape (C, H, W).
    # projected_pix: integer pixel indices of shape (N, 2) where N = X·Y·Z is the number of voxels at the target 3D scale.
    # fov_mask: boolean mask of shape (N, ); voxels outside the image FOV are masked and get zero features

    # output:
    # B - batch size
    # X, Y, Z - grid dimensions
    # C - channels
    # B,C,X,Y,Z

    def __init__(self, scene_size: Tuple[int, int, int], projection_scale: int):
        super().__init__()
        self.scene_size = scene_size
        self.projection_scale = projection_scale

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x2d, projected_pix, fov_mask = inputs
        B, C, H, W = x2d.shape
        N = projected_pix.shape[1]

        src = x2d.view(B, C, H * W)                                    # (B, C, HW)
        zero_col = torch.zeros(B, C, 1, dtype=src.dtype, device=src.device)
        src_padded = torch.cat([src, zero_col], dim=2)                  # (B, C, HW+1)

        # Compute flattened image indices from (x,y); mask out-of-FOV to HW (the padded zero)
        pix_x, pix_y = projected_pix[..., 0], projected_pix[..., 1]                                    # (B, N)
        img_indices = pix_y * W + pix_x                                 # (B, N)
        img_indices = img_indices.masked_fill(~fov_mask, H * W)         # (B, N)
        img_indices = img_indices.long()

        # Gather features: expand indices across channel dim
        gather_idx = img_indices.unsqueeze(1).expand(B, C, N)           # (B, C, N)
        gathered = torch.gather(src_padded, dim=2, index=gather_idx)    # (B, C, N)

        # Reshape to (B, C, X, Y, Z)
        X = self.scene_size[0] // self.projection_scale
        Y = self.scene_size[1] // self.projection_scale
        Z = self.scene_size[2] // self.projection_scale
        assert X * Y * Z == N, f"N={N} must equal X*Y*Z={X*Y*Z} for the target scale."

        x3d = gathered.view(B, C, X, Y, Z)
        return x3d


class UNet3D(nn.Module):
    def __init__(
        self,
        config,
        norm_layer = nn.BatchNorm3d,
        context_prior = True,
        bn_momentum = 0.1,
    ):
        super().__init__()
        projection_scale = config.minecraft.projection_scale
        scene_size = config.minecraft.scene_size
        feature = config.minecraft.prediction_head.feature_size
        dilations = config.minecraft.prediction_head.dilations

        size_l1 = (
            int(scene_size[0] / projection_scale),
            int(scene_size[1] / projection_scale),
            int(scene_size[2] / projection_scale),
        )
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        self.process_l1 = nn.Sequential(
            Process(feature, norm_layer, bn_momentum, dilations),
            Downsample(feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(feature * 2, norm_layer, bn_momentum, dilations),
            Downsample(feature * 2, norm_layer, bn_momentum),
        )

        self.up_13_l2 = Upsample(feature * 4, feature * 2, norm_layer, bn_momentum)
        self.up_12_l1 = Upsample(feature * 2, feature, norm_layer, bn_momentum)
        # self.up_l1_lfull = Upsample(feature, feature // 2, norm_layer, bn_momentum)
        self.up_l1_lfull = nn.Sequential(
            nn.Conv3d(feature, feature // 2, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(feature // 2, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv3d(feature // 2, feature // 2, kernel_size=1, padding=0, stride=1)

        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(feature * 4, size_l3, bn_momentum=bn_momentum)

    def forward(self, x):
        x3d_l1 = x
        x3d_l2 = self.process_l1(x3d_l1)
        x3d_l3 = self.process_l2(x3d_l2)

        if self.context_prior:
            x3d_l3 = self.CP_mega_voxels(x3d_l3)["x"]

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        return self.out_conv(x3d_up_lfull)   # (B, out_channels, X, Y, Z)



class SegmentationHead(nn.Module):

    def __init__(self, config, classes):
        super().__init__()
        dilations = config.minecraft.prediction_head.dilations
        c_in = config.minecraft.prediction_head.feature_size // 2

        # ASPP Block
        self.conv_list = dilations
        self.conv1 = nn.ModuleList(
            [nn.Conv3d(c_in, c_in, kernel_size=3, padding=d, dilation=d, bias=False)
             for d in dilations]
        )
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(c_in) for _ in dilations])

        self.conv2 = nn.ModuleList(
            [nn.Conv3d(c_in, c_in, kernel_size=3, padding=d, dilation=d, bias=False)
             for d in dilations]
        )
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(c_in) for _ in dilations])

        self.relu = nn.ReLU(inplace=True)

        out_ch = classes # len(classes) + sum(classes)
        self.conv_classes = nn.Conv3d(c_in, out_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x)))))
        x = self.relu(y + x)
        return self.conv_classes(x)
