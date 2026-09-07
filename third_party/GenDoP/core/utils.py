import math
import torch
import numpy as np
import trimesh
import megfile
from core.options import Options
import logging


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = torch.sqrt(tr) / 2.0
            x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
            y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
            z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
        elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
            S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
            r = (M[ 2, 1] - M[ 1, 2]) / S
            x = 0.25 * S
            y = (M[ 0, 1] + M[ 1, 0]) / S 
            z = (M[ 0, 2] + M[ 2, 0]) / S 
        elif M[ 1, 1] > M[ 2, 2]: 
            S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
            r = (M[ 0, 2] - M[ 2, 0]) / S
            x = (M[ 0, 1] + M[ 1, 0]) / S
            y = 0.25 * S
            z = (M[ 1, 2] + M[ 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
            r = (M[ 1, 0] - M[ 0, 1]) / S
            x = (M[ 0, 2] + M[ 2, 0]) / S
            y = (M[ 1, 2] + M[ 2, 1]) / S
            z = 0.25 * S
        Q = torch.stack([r, x, y, z], dim=-1)
        Qs += [Q]

    return torch.stack(Qs, dim=0).reshape(*prefix_shape, 4)

def quaternion_slerp(
    q0, q1, fraction, spin: int = 0, shortestpath: bool = True
):
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    d = (q0 * q1).sum(-1)
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[d < 0.0] = q1[d < 0.0]

    d = d.clamp(0, 1.0)

    angle = torch.acos(d) + spin * math.pi
    isin = 1.0 / (torch.sin(angle)+ 1e-10)
    q0_ = q0 * torch.sin((1.0 - fraction) * angle) * isin
    q1_ = q1 * torch.sin(fraction * angle) * isin

    q = q0_ + q1_
    q[angle < 1e-5, :] = q0

    return q


def sample_from_two_pose(pose_a, pose_b, fraction, noise_strengths=[0, 0]):
    """
    Args:
        pose_a: first pose
        pose_b: second pose
        fraction
    """
    def is_valid_rotation_matrix(matrix):
        should_be_identity = torch.matmul(matrix, matrix.transpose(-1, -2))
        identity = torch.eye(3, device=matrix.device).expand_as(should_be_identity)
        return torch.allclose(should_be_identity, identity, atol=1e-6) and torch.allclose(torch.det(matrix), torch.ones_like(matrix))

    quat_a = matrix_to_quaternion(pose_a[..., :3, :3])
    quat_b = matrix_to_quaternion(pose_b[..., :3, :3])
    dot = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    quat_b = torch.where(dot < 0, -quat_b, quat_b)

    cos_theta = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    slerp_condition = cos_theta.abs() < 0.9995
    slerp_quat = quaternion_slerp(quat_a, quat_b, fraction)
    lerp_quat = (1 - fraction) * quat_a + fraction * quat_b
    lerp_quat = lerp_quat / lerp_quat.norm(dim=-1, keepdim=True)
    quaternion = torch.where(slerp_condition, slerp_quat, lerp_quat)
    
    quaternion = torch.nn.functional.normalize(quaternion + torch.randn_like(quaternion) * noise_strengths[0], dim=-1)

    R = quaternion_to_matrix(quaternion)
    T = (1 - fraction) * pose_a[..., :3, 3] + fraction * pose_b[..., :3, 3]
    T = T + torch.randn_like(T) * noise_strengths[1]

    new_pose = pose_a.clone()
    new_pose[..., :3, :3] = R
    new_pose[..., :3, 3] = T
    
    # assert is_valid_rotation_matrix(R), "Invalid rotation matrix"
    if not is_valid_rotation_matrix(R):
        print(new_pose.shape)
        print("Invalid rotation matrix")
        new_pose[..., :3, :3] = torch.eye(3)

    return new_pose

def sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0, 0, 0, 0]):
    B, N, C = dense_cameras.shape
    B, M = t.shape
    
    left = torch.floor(t * (N-1)).long().clamp(0, N-2)
    right = left + 1
    fraction = t * (N-1) - left
    a = torch.gather(dense_cameras, 1, left[..., None].repeat(1, 1, C))
    b = torch.gather(dense_cameras, 1, right[..., None].repeat(1, 1, C))

    new_pose = sample_from_two_pose(a[:, :, :12].reshape(B, M, 3, 4),
                                    b[:, :, :12].reshape(B, M, 3, 4), fraction, noise_strengths=noise_strengths[:2])

    new_ins = (1 - fraction) * a[:, :, 12:] + fraction * b[:, :, 12:]

    return torch.cat([new_pose.reshape(B, M, 12), new_ins], dim=2)  
    
    
def camera_to_token(cameras):
    B, N, _ = cameras.shape

    RT = cameras[:, :, :12].reshape(B, N, 3, 4)
    # rotation
    rotation = matrix_to_quaternion(RT[:, :, :, :3])
    # translation
    translation = RT[:, :, :, 3]
    # fx, fy, cx, cy
    intrinsics = torch.stack([cameras[:, :, 12] / cameras[:, :, 16],
                                cameras[:, :, 13] / cameras[:, :, 17],
                                cameras[:, :, 14] / cameras[:, :, 16],
                                cameras[:, :, 15] / cameras[:, :, 17]], dim=2)

    return torch.cat([rotation, translation, intrinsics], dim=2)


def camera_to_token_single(cameras):
    N, _ = cameras.shape

    RT = cameras[:, :12].reshape(N, 3, 4)
    # rotation
    rotation = matrix_to_quaternion(RT[:, :, :3])
    # translation
    translation = RT[:, :, 3]
    # fx, fy, cx, cy
    intrinsics = torch.stack([cameras[:, 12] / cameras[:, 16],
                                cameras[:, 13] / cameras[:, 17]], dim=1)

    return torch.cat([rotation, translation, intrinsics], dim=1)

def token_to_camera(tokens, W, H):
    B, N, _ = tokens.shape

    R = quaternion_to_matrix(tokens[:, :, :4]) # B, N, 3, 3
    T = tokens[:, :, 4:7].reshape(B, N, 3, 1) # B, N, 3, 1

    RT = torch.cat([R, T], dim=3).reshape(B, N, 12)

    intrinsics = torch.stack([tokens[:, :, 7] * W,
                                tokens[:, :, 8] * H,
                                torch.full((B, N), fill_value=W/2),
                                torch.full((B, N), fill_value=H/2),
                                torch.full((B, N), fill_value=W),
                                torch.full((B, N), fill_value=H),
                                ], dim=2)

    return torch.cat([RT, intrinsics], dim=2)

def init_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # write to file
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # print to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger
    
def monkey_patch_transformers():
    import torch
    import math
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, ExponentialDecayLengthPenalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        # MODIFICATION: use input_ids.shape[0] instead of -1 to avoid confusion
        for batch_id, beam_sent in enumerate(input_ids.view(input_ids.shape[0], self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed
    
    PrefixConstrainedLogitsProcessor.__call__ = __call__
    print(f'[INFO] monkey patched PrefixConstrainedLogitsProcessor.__call__')