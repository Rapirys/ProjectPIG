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

def intrinsics_from_fov(W: int, H: int, hfov_deg: float, vfov_deg: float,
                        device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a 3x3 pinhole K from FOVs (OpenCV-style: cx=W/2, cy=H/2)."""
    #TODO rewrite with PyTorch3D or scypy
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)
    fx = W / (2.0 * math.tan(hfov / 2.0))
    fy = H / (2.0 * math.tan(vfov / 2.0))
    cx = W / 2.0
    cy = H / 2.0
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    return K


def extrinsics_from_player_position(pose: torch.Tensor, device = None) -> torch.Tensor:
    """
    pose: [B,5] or [5] = [x, y, z, yaw_deg(clockwise from south), pitch_deg(down+)]
    Returns: cam_E [B,4,4] world→camera. Mirrors the NumPy reference op-by-op.
    """
    # force identical dtype/op order as NumPy reference
    if pose.ndim == 1:
        pose = pose[None, :]
    P = pose.to(dtype=torch.float32, device=device)             # NumPy path uses float32
    x, y, z, yaw_deg, pitch_deg = P.T

    # angles
    yaw   = torch.deg2rad(-yaw_deg)              # invert -> CCW+
    pitch = torch.deg2rad(pitch_deg)

    cy, sy = torch.cos(yaw),   torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)

    B = P.shape[0]
    Ry = torch.zeros((B, 3, 3), dtype=torch.float32, device=P.device)
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy

    Rx = torch.zeros((B, 3, 3), dtype=torch.float32, device=P.device)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cp
    Rx[:, 1, 2] = -sp
    Rx[:, 2, 1] = sp
    Rx[:, 2, 2] = cp

    # match numpy einsum path exactly
    Rcw = torch.einsum('bij,bjk->bik', Ry, Rx)   # camera→world
    Rwc = Rcw.transpose(1, 2)                    # world→camera

    t   = torch.stack([x, y, z], dim=-1)         # (B,3)
    twc = -torch.einsum('bij,bj->bi', Rwc, t)    # world→camera translation

    E = torch.zeros((B, 4, 4), dtype=torch.float32, device=P.device)
    E[:, :3, :3] = Rwc
    E[:, :3,  3] = twc
    E[:,  3,  3] = 1.0

    return E


def extract_camera_position_from_info(info) -> np.ndarray:
    return np.asarray([info["xpos"], info["ypos"] + 1.62, info["zpos"], info["yaw"], info["pitch"]], dtype=np.float32)


def build_globalid_to_blockid_lut(block_state_registry, device=None):
    max_gid = max(m['global_id']
                  for b in block_state_registry['blocks']
                  for m in b['metas'])
    lut = torch.full((max_gid + 1,), -1, dtype=torch.long, device=device)

    for b in block_state_registry['blocks']:
        bid = b['block_id']
        for m in b['metas']:
            lut[m['global_id']] = bid

    return lut