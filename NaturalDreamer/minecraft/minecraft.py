import torch
from torch import nn

from NaturalDreamer.minecraft.monoscene import FLoSP, SegmentationHead, UNet3D
from NaturalDreamer.minecraft.utils import (
    vox2pix,
    intrinsics_from_fov,
    extrinsics_from_player_position,
)
from NaturalDreamer.networks import DecoderConv


class MinecraftSegmentationHead(nn.Module):

    #Input:
    # 1 d vector of features

    # Output:
    # B - batch size
    # X, Y, Z - grid dimensions
    # B_c - number of block clases
    # number of block states (
    #predicrs grid B,X,Y,Z,B_c,16

    def __init__(self, config, device, input_size, observationShape, classes):
        super().__init__()

        self.device = device
        self.img_W = observationShape[1]
        self.img_H = observationShape[2]
        self.fov = config.minecraft.fov
        self.cam_K = intrinsics_from_fov(self.img_W, self.img_H, self.fov, self.fov, device)


        self.config = config
        self.scene_size = config.minecraft.scene_size
        self.projection_scale = config.minecraft.projection_scale

       # 2d features
        self.decoder2d = DecoderConv(
            input_size, [config.minecraft.prediction_head.feature_size,self.img_W, self.img_H], config.decoder #TODO feature_size is under questions, especially if we increase the resolution
        ).to(self.device)
        self.flosp = FLoSP(self.scene_size, self.projection_scale).to(self.device)
        self.net_3d_decoder = UNet3D(config).to(self.device)
        self.segmentationHead = SegmentationHead(config, classes).to(self.device)

        self.network = nn.Sequential(
            self.flosp,
            self.net_3d_decoder,
            self.segmentationHead,
            # Minecraft completion head
        )

    def forward(self, x, camera_position, grid_origin):
        cam_E = extrinsics_from_player_position(camera_position)
        cam_E = torch.from_numpy(cam_E).to(self.device)
        projected_pix, fov_mask, _, _ = vox2pix(cam_E, self.cam_K, grid_origin, self.img_W, self.img_H, self.scene_size)
        x = self.decoder2d(x)
        return self.network((x, projected_pix, fov_mask))

def get_classes(block_state_registry):
    blocks =block_state_registry["blocks"]
    blocks.sort(key=lambda b: b["block_id"])
    return [len(block['metas']) for block in blocks]
