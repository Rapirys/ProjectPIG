import spconv.pytorch as spconv
import torch
from torch import nn

from minecraftscc.sparse.models import SparseFLoSPFromMDN
from minecraftscc.sparse.unet3d import UNet3D, SegmentationHead

MARGIN = 32


class SparseMinecraftSegmentationHead(nn.Module):
    """
    Sparse prediction head:

      DepthMDNHead (outside) -> SparseFLoSPFromMDN -> UNet3D -> SegmentationHead

    Input:
        features_2d:        [B, C, H, W]
        mixture_weights:    [B, K, H, W]
        component_means:    [B, K, H, W]
        component_scales:   [B, K, H, W]
        camera_position:    [B, 3]
        model_view_metrix:  [B, 4, 4]
        projection_metrix:  [4, 4]
        target_coords:      [Q, 4] world coords (b,x,y,z)

    Output:
        logits: torch.Tensor [Q, classes]
    """

    def __init__(self, config, device, classes: int):
        super().__init__()
        self.device = device
        self.config = config

        self.flosp_sparse = SparseFLoSPFromMDN().to(self.device)
        self.net_3d_decoder = UNet3D(config).to(self.device)
        self.segmentation_head = SegmentationHead(config, classes).to(self.device)

    def forward(
            self,
            features_2d: torch.Tensor,  # [B, C, H, W]
            mixture_weights: torch.Tensor,  # [B, K, H, W]
            component_means: torch.Tensor,  # [B, K, H, W]
            component_scales: torch.Tensor,  # [B, K, H, W]
            camera_position: torch.Tensor,  # [B, 3]
            model_view_metrix: torch.Tensor,  # [B, 4, 4]
            projection_metrix: torch.Tensor,  # [4, 4]
            target_coords: torch.Tensor,        # [Q, 4] world coords (b,x,y,z)
    ) -> torch.Tensor:
        # 1) Sample depth and lift to sparse 3D in camera-local frame (camera at origin)
        origin, spatial_shape = self._world_origin_and_shape(camera_position)

        # targets are in world frame -> convert to local frame used by SparseConvTensor
        target_coords_local = target_coords.to(device=camera_position.device, dtype=torch.int32).clone()
        b = target_coords_local[:, 0].to(torch.long)
        target_coords_local[:, 1:] = target_coords_local[:, 1:] - origin[b]

        coords, feats = self.flosp_sparse(
            mixture_weights,
            component_means,
            component_scales,
            features_2d,
        camera_position,
            model_view_metrix,
            projection_metrix,
        )  # coords: [M,4] (b,x,y,z), feats: [M,C]

        # 2) Drop coords outside spatial_shape before building SparseConvTensor
        b_coords = coords[:, 0].to(torch.long)
        xyz = coords[:, 1:].to(torch.long)
        xyz = xyz - origin[b_coords]

        in_bounds = (
            (xyz[:, 0] >= 0) & (xyz[:, 0] < spatial_shape[0]) &
            (xyz[:, 1] >= 0) & (xyz[:, 1] < spatial_shape[1]) &
            (xyz[:, 2] >= 0) & (xyz[:, 2] < spatial_shape[2])
        )
        coords = coords[in_bounds]
        coords[:, 1:] = xyz[in_bounds]
        feats = feats[in_bounds]

        x_sparse = spconv.SparseConvTensor(feats, coords, spatial_shape, camera_position.shape[0])
        # 3) 3D UNet decoder
        x_3d = self.net_3d_decoder(x_sparse)

        # 4) Segmentation head (returns logits only at target coords)
        logits = self.segmentation_head(x_3d, target_coords_local)
        return logits

    def _world_origin_and_shape(self, camera_position: torch.Tensor):
        render_dist_blocks = self.config.minecraft["render_distance"] * 16 + MARGIN  # int
        player_x, player_z = camera_position[:, 0], camera_position[:, 2]
        x0 = torch.floor(player_x - render_dist_blocks).to(torch.int32)
        z0 = torch.floor(player_z - render_dist_blocks).to(torch.int32)
        y0, y1 = 0, 256

        origin = torch.stack([x0, torch.full_like(x0, y0), z0], dim=1)  # [B,3]
        spatial_shape = [2 * render_dist_blocks, y1 - y0, 2 * render_dist_blocks]
        return origin, spatial_shape

