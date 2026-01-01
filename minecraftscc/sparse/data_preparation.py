from __future__ import annotations

import numpy as np
import torch
from typing import Tuple

@torch.no_grad()
def backproject_visible_blocks_from_depth(
    depth_camera_z: torch.Tensor,   # [B,H,W] float32, camera-Z depth (+Z forward)
    camera_position: torch.Tensor,  # [B,3] or [3]
    model_view_metrix: torch.Tensor,
    projection_metrix: torch.Tensor,
    image_width: int,
    image_height: int
) -> torch.Tensor:
    """
    Returns MinkowskiEngine coords: int32 [N,4] = (batch, x, y, z) of first-hit blocks.
    Pixel centers are used: (u+0.5, v+0.5).
    """
    device = depth_camera_z.device
    P = projection_metrix
    world_to_camera = model_view_metrix
    camera_to_world = torch.linalg.inv(world_to_camera)

    P00 = P[0, 0]
    P11 = P[1, 1]

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(image_height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(image_width,  device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )  # [H,W], [H,W]

    depth = depth_camera_z.to(torch.float32)  # [B,H,W]
    valid = torch.isfinite(depth) & (depth > 1e-12)

    x_ndc = (pixel_x / float(image_width)) * 2.0 - 1.0          # [H,W]
    y_ndc = 1.0 - (pixel_y / float(image_height)) * 2.0         # [H,W]

    # --- OpenGL camera space (forward is -Z)
    z_cam = -depth                                              # [B,H,W]
    x_cam = -x_ndc[None] * z_cam / P00                           # [B,H,W]
    y_cam = -y_ndc[None] * z_cam / P11                           # [B,H,W]
    ones = torch.ones_like(z_cam)

    points_camera_h = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1).view(depth.shape[0], -1, 4)  # [B,HW,4]
    valid_flat = valid.view(depth.shape[0], -1)

    points_world = points_camera_h @ camera_to_world.transpose(1, 2)          # [B,HW,4]
    points_world_xyz = points_world[..., :3]                                  # [B,HW,3]

    if camera_position.dim() == 1:
        camera_position = camera_position.unsqueeze(0)                        # [1,3]
    points_world_xyz = points_world_xyz + camera_position[:, None, :]         # [B,HW,3]

    camera_world_xyz = camera_to_world[:, :3, 3].unsqueeze(1)                 # [B,1,3]
    points_world_xyz = points_world_xyz - 1e-4 * torch.nn.functional.normalize(points_world_xyz - camera_world_xyz, dim=2) #Step Bias along the viewing ray
    block_xyz = torch.floor(points_world_xyz).to(torch.int32)                 # [B,HW,3]
    batch_ids = torch.arange(depth.shape[0], device=device, dtype=torch.int32)[:, None].expand_as(valid_flat)
    coords = torch.cat([batch_ids[valid_flat][:, None], block_xyz[valid_flat]], dim=1)  # [N,4]

    return torch.unique(coords, dim=0)


@torch.no_grad()
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


def visible_blocks_with_halo_numpy(
    depth_camera_z: torch.Tensor,   # [B,H,W]
    camera_position: torch.Tensor,
    model_view_metrix: torch.Tensor,
    projection_metrix: torch.Tensor,
    image_width: int,
    image_height: int,
    halo_size: Tuple[int, int, int] = (3, 3, 3),
) -> Tuple[np.ndarray, np.ndarray]:
    B = depth_camera_z.shape[0]
    if model_view_metrix.ndim == 2:
        model_view_metrix = model_view_metrix.unsqueeze(0).expand(B, -1, -1).contiguous()

    visible_coords = backproject_visible_blocks_from_depth(
        depth_camera_z, camera_position, model_view_metrix, projection_metrix, image_width, image_height,
    )
    halo_coords = halo_around_visible_blocks_torch(visible_coords, halo_size=halo_size)  # torch [N*Q,4]

    all_coords, inv = torch.unique(torch.cat([visible_coords, halo_coords], dim=0), dim=0, return_inverse=True)

    n_vis = visible_coords.shape[0]
    direct_mask = torch.zeros(all_coords.shape[0], device=all_coords.device, dtype=torch.bool)
    direct_mask[inv[:n_vis]] = True

    return (
        all_coords.cpu().numpy().astype(np.int32, copy=False),
        direct_mask.cpu().numpy()
    )

# -0.83867055, -0.028504208, -0.5438926, 0.0, 0.0, 0.9986295, -0.05233596, 0.0, 0.54463905, -0.043892626, -0.83752114, 0.0, 0.0, -1.5378894, 0.13059738, 1.0