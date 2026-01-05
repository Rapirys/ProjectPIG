import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from torch import nn


class SparseBN(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return x.replace_feature(self.bn(x.features))


class SparseReLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return x.replace_feature(F.relu(x.features, inplace=self.inplace))

class SpconvBottleneck(nn.Module):
    """
    Simple 3D bottleneck block for spconv:
      in -> 1x1 -> 3x3(dilated) -> 1x1 -> +res -> ReLU
    Channels are kept constant (no expansion).
    """

    def __init__(self, channels: int, bn_momentum: float, dilation: int = 1, idx_prefix: str = "proc"):
        super().__init__()
        mid = channels // 4 if channels >= 4 else channels

        self.conv1 = spconv.SubMConv3d(
            in_channels=channels,
            out_channels=mid,
            kernel_size=1,
            stride=1,
        )
        self.bn1 = SparseBN(mid, momentum=bn_momentum)
        self.relu1 = SparseReLU(inplace=True)

        self.conv2 = spconv.SubMConv3d(
            in_channels=mid,
            out_channels=mid,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            indice_key=f"{idx_prefix}_3x3_d{dilation}",
        )
        self.bn2 = SparseBN(mid, momentum=bn_momentum)
        self.relu2 = SparseReLU(inplace=True)

        self.conv3 = spconv.SubMConv3d(
            in_channels=mid,
            out_channels=channels,
            kernel_size=1,
            stride=1,
        )
        self.bn3 = SparseBN(channels, momentum=bn_momentum)
        self.relu_out = SparseReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # residual add
        out = out.replace_feature(out.features + identity.features)
        out = self.relu_out(out)
        return out

class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3], stage_prefix: str = "proc"):
        super().__init__()
        self.main = nn.Sequential(
            *[
                SpconvBottleneck(
                    channels=feature,
                    bn_momentum=bn_momentum,
                    dilation=d,
                    idx_prefix=f"{stage_prefix}_d{d}",
                )
                for d in dilations
            ]
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.main(x)



class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum, indice_key: str):
        super().__init__()
        self.main = nn.Sequential(
            spconv.SparseInverseConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                indice_key=indice_key,
            ),
            SparseBN(out_channels, momentum=bn_momentum),
            SparseReLU(inplace=True),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8, indice_key: str = "ds_l1_to_l2"):
        super().__init__()
        out_channels = feature * 2

        self.conv = spconv.SparseConv3d(
                in_channels=feature,
                out_channels=out_channels,
                kernel_size=2,
            stride=2,
            indice_key=indice_key,
        )
        self.bn = SparseBN(out_channels, momentum=bn_momentum)
        self.relu = SparseReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, planes, dilations_conv_list):
        super().__init__()
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                spconv.SubMConv3d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3,
                    padding=dil,
                    dilation=dil,
                    indice_key=f"cp_aspp_d{dil}",  # share with conv2 for reuse
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [SparseBN(planes) for _ in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                spconv.SubMConv3d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3,
                    padding=dil,
                    dilation=dil,
                    indice_key=f"cp_aspp_d{dil}",  # share with conv1 for reuse
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [SparseBN(planes) for _ in dilations_conv_list]
        )

        self.relu = SparseReLU(inplace=True)

    def forward(self, x_in: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y_i = self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
            y = y.replace_feature(y.features + y_i.features)

        y_res = y.replace_feature(y.features + x_in.features)
        x_out = self.relu(y_res)
        return x_out


class CPMegaVoxels(nn.Module): # TODO rename

    def __init__(self, feature, n_relations=4, bn_momentum=0.0003):
        super().__init__()
        self.n_relations = n_relations
        self.feature = feature

        self.aspp = ASPP(feature, [1, 2, 3])

        self.resize = nn.Sequential(
            spconv.SubMConv3d(
                in_channels=feature,
                out_channels=feature,
                kernel_size=1,
                stride=1,
                ),
            SparseBN(feature, momentum=bn_momentum),
            SparseReLU(inplace=True),
            Process(feature, nn.BatchNorm3d, bn_momentum, dilations=[1], stage_prefix="cp_proc"),
        )

    def forward(self, input: spconv.SparseConvTensor):
        x_agg = self.aspp(input)
        x = self.resize(x_agg)

        ret = {
            "P_logits": None,  # context prior logits omitted in sparse version
            "x": x,
        }
        return ret



class UNet3D(nn.Module):
    def __init__(
        self,
        config,
        norm_layer = nn.BatchNorm3d,
        context_prior = True,
        bn_momentum = 0.1,
    ):
        super().__init__()
        projection_scale = config.minecraft["projection_scale"]
        feature = config.minecraft["prediction_head"]["feature_size"]
        dilations = config.minecraft["prediction_head"]["dilations"]


        self.process_l1 = nn.Sequential(
            Process(feature, norm_layer, bn_momentum, dilations, stage_prefix="l1_proc"),
            Downsample(feature, norm_layer, bn_momentum, indice_key="ds_l1_to_l2"),
        )
        self.process_l2 = nn.Sequential(
            Process(feature * 2, norm_layer, bn_momentum, dilations, stage_prefix="l2_proc"),
            Downsample(feature * 2, norm_layer, bn_momentum, indice_key="ds_l2_to_l3"),
        )

        self.up_13_l2 = Upsample(feature * 4, feature * 2, norm_layer, bn_momentum, indice_key="ds_l2_to_l3")
        self.up_12_l1 = Upsample(feature * 2, feature, norm_layer, bn_momentum, indice_key="ds_l1_to_l2")

        self.up_l1_lfull = nn.Sequential(
            spconv.SubMConv3d(
                in_channels=feature,
                out_channels=feature // 2,
                kernel_size=1,
                stride=1,
            ),
            SparseBN(feature // 2, momentum=bn_momentum),
            SparseReLU(inplace=True),
        )
        self.out_conv = spconv.SubMConv3d(
            in_channels=feature // 2,
            out_channels=feature // 2,
            kernel_size=1,
            stride=1,
        )

        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(feature * 4, bn_momentum=bn_momentum)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x3d_l1 = x
        x3d_l2 = self.process_l1(x3d_l1)
        x3d_l3 = self.process_l2(x3d_l2)

        if self.context_prior:
            x3d_l3 = self.CP_mega_voxels(x3d_l3)["x"]

        x3d_up_l2 = self.up_13_l2(x3d_l3)
        x3d_up_l2 = x3d_up_l2.replace_feature(x3d_up_l2.features + x3d_l2.features)

        x3d_up_l1 = self.up_12_l1(x3d_up_l2)
        x3d_up_l1 = x3d_up_l1.replace_feature(x3d_up_l1.features + x3d_l1.features)

        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        out = self.out_conv(x3d_up_lfull)
        return out


class SegmentationHead(nn.Module):
    def __init__(self, config, classes: int):
        super().__init__()
        dilations = config.minecraft["prediction_head"]["dilations"]
        c_in = config.minecraft["prediction_head"]["feature_size"] // 2

        self.conv_list = dilations
        self.conv1 = nn.ModuleList(
            [
                spconv.SubMConv3d(
                    in_channels=c_in,
                    out_channels=c_in,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    indice_key=f"seg_d{d}",  # share with conv2 for reuse
                )
             for d in dilations]
        )
        self.bn1 = nn.ModuleList([SparseBN(c_in) for _ in dilations])

        self.conv2 = nn.ModuleList(
            [
                spconv.SubMConv3d(
                    in_channels=c_in,
                    out_channels=c_in,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    indice_key=f"seg_d{d}",  # share with conv1 for reuse
                )
                for d in dilations
            ]
        )
        self.bn2 = nn.ModuleList(
            [SparseBN(c_in) for _ in dilations]
        )

        self.relu = SparseReLU(inplace=True)

        # Dense classifier applied only at target coords.
        self.classifier = nn.Linear(c_in, classes)

    def forward(self, x: spconv.SparseConvTensor, target_coords: torch.Tensor) -> torch.Tensor:
        # target_coords: [Q,4] (b,x,y,z) in the same local frame as x.indices
        n = x.indices.shape[0]
        all_idx = torch.cat([x.indices, target_coords], dim=0)  # [n+Q,4]
        uniq_idx, inv = torch.unique(all_idx, dim=0, return_inverse=True)

        # Only original points contribute features; targets contribute implicit zeros.
        uniq_feat = x.features.new_zeros((uniq_idx.shape[0], x.features.shape[1]))
        uniq_feat.index_add_(0, inv[:n], x.features)
        target_rows = inv[n:]  # [Q]

        x = spconv.SparseConvTensor(uniq_feat, uniq_idx, x.spatial_shape, x.batch_size)

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x)))))
        for i in range(1, len(self.conv_list)):
            y_i = self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x)))))
            y = y.replace_feature(y.features + y_i.features)

        x_out = self.relu(y.replace_feature(y.features + x.features))
        return self.classifier(x_out.features[target_rows])  # [Q, classes]
