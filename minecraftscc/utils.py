import math

import numpy as np
import torch
#
# def extrinsics_from_player_position(pose: torch.Tensor, device = None) -> torch.Tensor:
#     """
#     pose: [B,5] or [5] = [x, y, z, yaw_deg(clockwise from south), pitch_deg(down+)]
#     Returns: cam_E [B,4,4] world→camera. Mirrors the NumPy reference op-by-op.
#     """
#     # force identical dtype/op order as NumPy reference
#     if pose.ndim == 1:
#         pose = pose[None, :]
#     P = pose.to(dtype=torch.float32, device=device)             # NumPy path uses float32
#     x, y, z, yaw_deg, pitch_deg = P.T
#
#     # angles
#     yaw   = torch.deg2rad(-yaw_deg)              # invert -> CCW+
#     pitch = torch.deg2rad(pitch_deg)
#
#     cy, sy = torch.cos(yaw),   torch.sin(yaw)
#     cp, sp = torch.cos(pitch), torch.sin(pitch)
#
#     B = P.shape[0]
#     Ry = torch.zeros((B, 3, 3), dtype=torch.float32, device=P.device)
#     Ry[:, 0, 0] = cy
#     Ry[:, 0, 2] = sy
#     Ry[:, 1, 1] = 1.0
#     Ry[:, 2, 0] = -sy
#     Ry[:, 2, 2] = cy
#
#     Rx = torch.zeros((B, 3, 3), dtype=torch.float32, device=P.device)
#     Rx[:, 0, 0] = 1.0
#     Rx[:, 1, 1] = cp
#     Rx[:, 1, 2] = -sp
#     Rx[:, 2, 1] = sp
#     Rx[:, 2, 2] = cp
#
#     # match numpy einsum path exactly
#     Rcw = torch.einsum('bij,bjk->bik', Ry, Rx)   # camera→world
#     Rwc = Rcw.transpose(1, 2)                    # world→camera
#
#     t   = torch.stack([x, y, z], dim=-1)         # (B,3)
#     twc = -torch.einsum('bij,bj->bi', Rwc, t)    # world→camera translation
#
#     E = torch.zeros((B, 4, 4), dtype=torch.float32, device=P.device)
#     E[:, :3, :3] = Rwc
#     E[:, :3,  3] = twc
#     E[:,  3,  3] = 1.0
#
#     return E

#
# def intrinsics_from_fov(W: int, H: int, hfov_deg: float, vfov_deg: float,
#                         device=None, dtype=torch.float32) -> torch.Tensor:
#     """Return a 3x3 pinhole K from FOVs (OpenCV-style: cx=W/2, cy=H/2)."""
#     hfov = math.radians(hfov_deg)
#     vfov = math.radians(vfov_deg)
#     fx = W / (2.0 * math.tan(hfov / 2.0))
#     fy = H / (2.0 * math.tan(vfov / 2.0))
#     cx = W / 2.0
#     cy = H / 2.0
#     K = torch.tensor([[fx, 0.0, cx],
#                       [0.0, fy, cy],
#                       [0.0, 0.0, 1.0]], device=device, dtype=dtype)
#     return K



def extract_camera_position_from_info(info) -> np.ndarray:
    return np.asarray([info["xEyesPos"], info["yEyesPos"], info["zEyesPos"]], dtype=np.float32)


def projection_matrices_from_info(info):
    return info["model_view_metrix"], info["projection_metrix"]


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


def get_classes(block_state_registry):
    blocks = block_state_registry["blocks"]
    blocks.sort(key=lambda b: b["block_id"])
    return [len(block['metas']) for block in blocks]
