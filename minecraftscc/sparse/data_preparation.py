from __future__ import annotations

import numpy as np
import torch
from typing import Tuple


def backproject_visible_points_from_depth(
    depth_camera_z: torch.Tensor,   # [B,H,W]
    camera_position: torch.Tensor,  # [B,3] or [3]
    model_view_metrix: torch.Tensor,
    projection_metrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (dense, all pixels):
        batch_ids:    [M]   int64, M = B*H*W
        points_world:  [M,3]  float32
    """
    device = depth_camera_z.device
    B, H, W = depth_camera_z.shape

    P = projection_metrix
    world_to_camera = model_view_metrix
    camera_to_world = torch.linalg.inv(world_to_camera)
    P00, P11 = P[:, 0, 0].view(B, 1, 1) , P[:, 1, 1].view(B, 1, 1)

    # integer pixel indices and centers
    pix_y_int = torch.arange(H, device=device, dtype=torch.long)
    pix_x_int = torch.arange(W, device=device, dtype=torch.long)
    pix_y, pix_x = torch.meshgrid(pix_y_int, pix_x_int, indexing="ij")      # [H,W],[H,W]

    pix_y_center = pix_y.to(torch.float32) + 0.5
    pix_x_center = pix_x.to(torch.float32) + 0.5

    depth = depth_camera_z.to(torch.float32)  # [B,H,W]
    x_ndc = (pix_x_center / float(W)) * 2.0 - 1.0
    y_ndc = 1.0 - (pix_y_center / float(H)) * 2.0


       # OpenGL camera space (forward is -Z)
    z_cam = -depth
    x_cam = -x_ndc[None] * z_cam / P00
    y_cam = -y_ndc[None] * z_cam / P11
    ones = torch.ones_like(z_cam)

    points_camera_h = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1).view(B, -1, 4)  # [B,HW,4]

    points_world = points_camera_h @ camera_to_world.transpose(1, 2)          # [B,HW,4]
    points_world_xyz = points_world[..., :3]                                  # [B,HW,3]

    if camera_position.dim() == 1:
        camera_position = camera_position.unsqueeze(0)                        # [1,3]
    points_world_xyz = points_world_xyz + camera_position[:, None, :]         # [B,HW,3]

    # flatten
    points_world_flat = points_world_xyz.reshape(B * H * W, 3)              # [M,3]
    batch_ids = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(H * W)

    return batch_ids, points_world_flat


def halo_around_visible_blocks_torch(
    visible_coords: torch.Tensor,                 # [N,4] int32 (batch,x,y,z)
    halo_size: Tuple[int, int, int] = (5, 5, 5) # (sx,sy,sz), includes center
) -> torch.Tensor:
    """Returns (possibly with duplicates) halo coords: int32 [N*Q,4] (batch,x,y,z)."""
    if visible_coords.numel() == 0:
        return visible_coords

    device = visible_coords.device
    sx, sy, sz = halo_size
    rx, ry, rz = sx // 2, sy // 2, sz // 2

    ox = torch.arange(-rx, rx + 1, device=device, dtype=torch.int32)
    oy = torch.arange(-ry, ry + 1, device=device, dtype=torch.int32)
    oz = torch.arange(-rz, rz + 1, device=device, dtype=torch.int32)
    offsets = torch.stack(torch.meshgrid(ox, oy, oz, indexing="ij"), dim=-1).view(-1, 3)  # [Q,3]

    batches = visible_coords[:, :1]   # [N,1]
    xyz = visible_coords[:, 1:] # [N,3]

    halo_xyz = xyz[:, None, :] + offsets[None, :, :]                 # [N,Q,3]
    halo_b = batches[:, None, :].expand(-1, offsets.shape[0], -1)          # [N,Q,1]
    halo_coords = torch.cat([halo_b, halo_xyz], dim=2).reshape(-1, 4) # [N*Q,4]
    return halo_coords

def backproject_visible_blocks_from_depth(
    depth_camera_z: torch.Tensor,   # [B,H,W]
    camera_position: torch.Tensor,  # [B,3] or [3]
    model_view_metrix: torch.Tensor,
    projection_metrix: torch.Tensor,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns MinkowskiEngine coords: int32 [N,4] = (batch, x, y, z) of first-hit blocks.
    Pixel centers are used: (u+0.5, v+0.5).
    """
    world_to_camera = model_view_metrix
    camera_to_world = torch.linalg.inv(world_to_camera)

    batch_ids, points_world_xyz = backproject_visible_points_from_depth(
        depth_camera_z,
        camera_position,
        model_view_metrix,
        projection_metrix
    )  # batch_ids:[M], points_world_xyz:[M,3]

    # ensure camera_position is [B,3]
    if camera_position.dim() == 1:
        camera_position = camera_position.unsqueeze(0).expand(world_to_camera.shape[0], -1)
    camera_pos_for_points = camera_position[batch_ids.long()]             # [M,3]

    camera_world_xyz = camera_to_world[:, :3, 3]                       # [B,3]
    camera_world_for_points = camera_world_xyz[batch_ids.long()]       # [M,3]

    # 1) go to local camera frame by subtracting global offset, then subtract local camera center
    directions = (points_world_xyz - camera_pos_for_points) - camera_world_for_points  # [M,3]
    directions = torch.nn.functional.normalize(directions, dim=1)
    points_world_xyz = points_world_xyz - 1e-4 * directions            # step bias

    block_xyz = torch.floor(points_world_xyz).to(torch.int32)          # [M,3]
    batch_ids_int = batch_ids.to(torch.int32)
    coords = torch.cat([batch_ids_int[:, None], block_xyz], dim=1)     # [M,4]

    return torch.unique(coords, dim=0, return_inverse=True)



@torch.no_grad()
def visible_blocks_with_halo_numpy(
    depth_camera_z: torch.Tensor,   # [B,H,W]
    camera_position: torch.Tensor,
    model_view_metrix: torch.Tensor,
    projection_metrix: torch.Tensor,
    halo_size: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[np.ndarray, np.ndarray]:
    B = depth_camera_z.shape[0]
    if model_view_metrix.ndim == 2:
        model_view_metrix = model_view_metrix.unsqueeze(0).expand(B, -1, -1).contiguous()
    if projection_metrix.ndim == 2:
        projection_metrix = projection_metrix.unsqueeze(0).expand(B, -1, -1).contiguous()

    visible_coords, _ = backproject_visible_blocks_from_depth(depth_camera_z, camera_position, model_view_metrix, projection_metrix)
    halo_coords = halo_around_visible_blocks_torch(visible_coords, halo_size=halo_size)  # torch [N*Q,4]

    all_coords, inv = torch.unique(torch.cat([visible_coords, halo_coords], dim=0), dim=0, return_inverse=True)

    n_vis = visible_coords.shape[0]
    direct_mask = torch.zeros(all_coords.shape[0], device=all_coords.device, dtype=torch.bool)
    direct_mask[inv[:n_vis]] = True

    return (
        all_coords.cpu().numpy().astype(np.int32, copy=False),
        direct_mask.cpu().numpy()
    )

def lift_to_3d_by_depth(
    depth_samples: torch.Tensor,       # [B, N, H, W]
    features: torch.Tensor,            # [B, C, H, W]
    camera_position: torch.Tensor,     # [B,3]
    model_view_metrix: torch.Tensor,   # [B,4,4]
    projection_metrix: torch.Tensor,   # [4,4]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse FLoSP-like projection.

    Args:
        depth_samples: (B, N, H, W) camera-Z depths.
        features:      (B, C, H, W) 2D features (shared across N per pixel).

    Returns:
        coords: (M, 4) int32 MinkowskiEngine coords (batch, x, y, z)
        feats:  (M, C) float32 features, summed over all samples that map to same coord
    """
    B, N, H, W = depth_samples.shape
    C = features.shape[1]

    depth_flat = depth_samples.view(B * N, H, W)                       # [B*N,H,W]
    camera_position_flat = camera_position.repeat_interleave(N, dim=0)  # [B*N,3]
    model_view_flat = model_view_metrix.repeat_interleave(N, dim=0)   # [B*N,4,4]

    coords, inv = backproject_visible_blocks_from_depth(
        depth_flat,
        camera_position_flat,
        model_view_flat,
        projection_metrix,
    )  # coords:[M,4], inv:[B*N*H*W]

    # features: [B,C,H,W] -> [B,H,W,C]
    feats_img = features.permute(0, 2, 3, 1)  # [B,H,W,C]
    feats_view = feats_img.unsqueeze(1).expand(B, N, H, W, C)
    feats_flat = feats_view.reshape(B * N * H * W, C)  # [B*N*H*W, C]

    M = coords.shape[0]
    feats = torch.zeros(M, C, device=features.device, dtype=features.dtype)
    feats.index_add_(0, inv, feats_flat)                                  # sum over duplicates

    return coords, feats
