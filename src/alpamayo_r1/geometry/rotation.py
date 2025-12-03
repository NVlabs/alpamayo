# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, TypeVar, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

TensorOrNDArray = TypeVar("TensorOrNDArray", torch.Tensor, np.ndarray)


def so3_to_yaw_torch(rot_mat: torch.Tensor) -> torch.Tensor:
    """Computes the yaw angle given an so3 rotation matrix (assumes that rotation is described in
    xyz order)

    Args:
        rot_mat (torch.Tensor): [..., 3,3]

    Returns:
        torch.Tensor: [...]
    """
    # phi is rotation about z, theta is rotation about y
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)


def so3_to_yaw_np(rot_mat: np.ndarray) -> np.ndarray:
    """Computes the yaw angle given an so3 rotation matrix (assumes that rotation is described in
    xyz order)

    Args:
        rot_mat (np.ndarray): [..., 3,3]

    Returns:
        np.ndarray: [...]
    """
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return np.arctan2(cos_th_sin_phi, cos_th_cos_phi)


def euler_2_so3(euler_angles: np.ndarray, degrees: bool = True, seq: str = "xyz") -> np.ndarray:
    """Converts the euler angles representation to the so3 rotation matrix
    Args:
        euler_angles (np.array): euler angles [n,3]
        degrees bool: True if angle is given in degrees else False
        seq string: sequence in which the euler angles are given

    Out:
        (np array): rotations given so3 matrix representation [n,3,3]
    """
    return (
        R.from_euler(seq=seq, angles=euler_angles, degrees=degrees).as_matrix().astype(np.float32)
    )


def angle_wrap(
    radians: TensorOrNDArray,
) -> TensorOrNDArray:
    """This function wraps angles to lie within [-pi, pi).

    Args:
        radians (np.ndarray): The input array of angles (in radians).

    Returns:
        np.ndarray: Wrapped angles that lie within [-pi, pi).
    """
    return (radians + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(angle: Union[float, np.ndarray]) -> np.ndarray:
    """Creates one or many 2D rotation matrices.

    Args:
        angle (Union[float, np.ndarray]): The angle to rotate points by.
            if float, returns 2x2 matrix
            if np.ndarray, expects shape [...], and returns [...,2,2] array

    Returns:
        np.ndarray: The 2x2 rotation matri(x/ces).
    """
    batch_dims = 0
    if isinstance(angle, np.ndarray):
        batch_dims = angle.ndim

    rotmat: np.ndarray = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    return rotmat.transpose(*np.arange(2, batch_dims + 2), 0, 1)


def rotation_matrix_torch(angle: torch.Tensor) -> torch.Tensor:
    """Creates one or many 2D rotation matrices.

    Args:
        angle (torch.Tensor): The angle to rotate points by. Size: [...].

    Returns:
        torch.Tensor: The 2x2 rotation matri(x/ces). Size: [..., 2, 2].
    """
    rotmat: torch.Tensor = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )
    return rotmat


def transform_coords_2d_np(
    coords: np.ndarray,
    offset: Optional[np.ndarray] = None,
    angle: Optional[np.ndarray] = None,
    rot_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Args:
        coords (np.ndarray): [..., 2] coordinates
        offset (Optional[np.ndarray], optional): [..., 2] offset to translate. Defaults to None.
        angle (Optional[np.ndarray], optional): [...] angle to rotate by. Defaults to None.
        rot_mat (Optional[np.ndarray], optional): [..., 2,2] rotation matrix to apply. Defaults to None.
            If rot_mat is given, angle is ignored.

    Returns:
        np.ndarray: transformed coords
    """
    if rot_mat is None and angle is not None:
        rot_mat = rotation_matrix(angle)

    if rot_mat is not None:
        coords = np.einsum("...ij,...j->...i", rot_mat, coords)

    if offset is not None:
        coords += offset

    return coords


def stable_gramschmidt(M: torch.Tensor) -> torch.Tensor:
    """Orthonormalize two 3D vectors using a stable Gram-Schmidt step.

    Args:
        M: Tensor of shape (..., 3, 2) with vectors (x, y).

    Returns:
        Tensor of shape (..., 3, 3) with orthonormal (x, y, x×y).
    """
    EPS = 1e-7

    x = M[..., 0]
    y = M[..., 1]
    x = x / torch.clamp_min(torch.norm(x, dim=-1, keepdim=True), EPS)
    y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = y / torch.clamp_min(torch.norm(y, dim=-1, keepdim=True), EPS)
    z = torch.cross(x, y, dim=-1)
    R = torch.stack((x, y, z), dim=-1)
    return R


def rot_3d_to_2d(rot):
    """Converts a 3D rotation matrix to a 2D rotation matrix by taking the x and y axes of the 3D
    rotation matrix, projecting them to xy plan, and performing gram-schmidt orthogonalization.

    Args:
        rot (torch.Tensor): The 3D rotation matrix to convert.

    Returns:
        torch.Tensor: The 2D rotation matrix.
    """
    xu = rot[..., :2, 0]
    yu = rot[..., :2, 1]
    EPS = 1e-6
    # gram-schmidt
    xu = xu / (torch.norm(xu, dim=-1, keepdim=True) + EPS)
    yu = yu - torch.sum(xu * yu, dim=-1, keepdim=True) * xu
    yu = yu / (torch.norm(yu, dim=-1, keepdim=True) + EPS)
    return torch.stack((xu, yu), dim=-1)


def rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    """Converts a 2D rotation matrix to a 3D rotation matrix assuming flat xy plane.

    Args:
        rot (torch.Tensor): The 2D rotation matrix to convert.

    Returns:
        torch.Tensor: The 3D rotation matrix.
    """
    rot = torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device).repeat(rot.shape[:-2] + (1, 1)),
        ],
        dim=-2,
    )
    return rot


def ratan2(s, c, eps=1e-4):
    """Robust arctan2 for pytorch
    torch.arctan2(0,0)=nan, this function avoids the nan situation and returns ratan2(0,0)=0
    """
    sign = (c >= 0).float() * 2 - 1
    eps = eps * (c.abs() < eps).type(c.dtype) * sign
    return torch.arctan2(s, c + eps)


def round_2pi(x: np.ndarray) -> np.ndarray:
    """Normalize angles to the range [-pi, pi].

    Args:
        x: Angle(s) in radians, can be numpy array or torch tensor

    Returns:
        Normalized angle(s) in the range [-pi, pi], same type as input
    """
    return np.atan2(np.sin(x), np.cos(x))


def round_2pi_torch(x: torch.Tensor) -> torch.Tensor:
    """Normalize angles to the range [-pi, pi] in torch.

    Args:
        x: Angle(s) in radians, can be numpy array or torch tensor

    Returns:
        Normalized angle(s) in the range [-pi, pi], same type as input
    """
    return torch.atan2(torch.sin(x), torch.cos(x))
