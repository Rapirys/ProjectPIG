import math

import numpy as np
import torch
from typing import Tuple


def make_voxel_centers(vox_origin: torch.Tensor,
                       scene_size: Tuple[float, float, float],
                       voxel_size = 1, device=None, dtype=torch.float32):

    device = device if device is not None else vox_origin.device
    scene_size_t = torch.tensor(scene_size, device=device, dtype=dtype)

    vol_dim = torch.ceil(scene_size_t / voxel_size).to(dtype=torch.int64)  # (3,)
    nx, ny, nz = int(vol_dim[0]), int(vol_dim[1]), int(vol_dim[2])

    grid = torch.stack(torch.meshgrid(
        torch.arange(nx, device=device),
        torch.arange(ny, device=device),
        torch.arange(nz, device=device),
        indexing='ij'
    ), dim=-1).to(dtype).reshape(-1, 3)

    centers = vox_origin.to(dtype)[:, None, :] + (grid[None, :, :] + 0.5) * float(voxel_size)
    return centers, vol_dim

@torch.no_grad()
def vox2pix(
    cam_E: torch.Tensor,  # (B,4,4) world-> , camera pose
    cam_K: torch.Tensor,  # (3,3) camera intrinsics
    vox_origin: torch.Tensor,  # (3,) min corner (x_min,y_min,z_min), tensor or tuple/list
    img_W: int, img_H: int,  # ints
    scene_size: Tuple[float, float, float],  # (size_x, size_y, size_z) meters
    device=None, dtype=torch.float32):
    """
    Returns:
      projected_pix: (B, N, 2)  int64   (u,v) pixel indices (rounded)
      fov_mask:      (B, N)     bool
      pix_z:         (B, N)     float   depth in camera-Z
    Assumes camera coords with +Z forward; FOV mask requires Z>0.
    """
    assert cam_E.dim() == 3 and cam_E.shape[-2:] == (4, 4), "cam_E must be (B,4,4)"
    B = cam_E.shape[0]
    device = device or cam_E.device
    cam_K = cam_K.expand(B, -1, -1)

    # centers_world voxel centers (N,3) in WORLD, plus dims
    centers_world, vol_dim = make_voxel_centers(vox_origin, scene_size, device=device, dtype=dtype)  # (N,3)
    N = centers_world.shape[1]

    # 1) World → Camera (homogeneous coordinates)
    ones_col = torch.ones((B, N, 1), device=device, dtype=dtype)
    points_world_h = torch.cat([centers_world, ones_col], dim=2)

    # Right-multiply row vectors by cam_E^T
    points_camera_h = torch.matmul(points_world_h, cam_E.transpose(1, 2))     # (B, N, 4)

    # 2) Camera → Pixel (pinhole projection)
    camera_xyz = points_camera_h[..., :3]
    pixel_h = camera_xyz @ cam_K.transpose(1, 2)  # (B,N,3) = [u_h, v_h, w_h]
    w = pixel_h[..., 2].clamp_min_(1e-8)  # (B,N)
    pixel_u = pixel_h[..., 0] / w  # (B,N)
    pixel_v = pixel_h[..., 1] / w  # (B,N)

    # 3) Discrete pixels, FOV mask, and depth output
    pixel_u_rounded = torch.round(pixel_u)
    pixel_v_rounded = torch.round(pixel_v)

    cam_z = camera_xyz[..., 2]
    visible_mask = (
        (pixel_u_rounded >= 0) & (pixel_u_rounded < img_W) &
        (pixel_v_rounded >= 0) & (pixel_v_rounded < img_H) &
        (cam_z > 1e-6)
    )

    projected_pix = torch.stack([pixel_u_rounded, pixel_v_rounded], dim=-1).to(torch.long)  # (B, N, 2)

    return projected_pix, visible_mask, cam_z, vol_dim

