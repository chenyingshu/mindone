# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Adapted from https://github.com/3DTopia/OpenLRM/blob/main/openlrm/models/rendering/utils/ray_sampler.py
# Modified by Zexin He in 2023-2024.
# The modifications are subject to the same license as the original.


"""
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import mindspore as ms
from mindspore import mint, nn, ops

from . import MeshGrid


class RaySampler(nn.Cell):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = (
            None,
            None,
            None,
            None,
            None,
        )
        self.mesh_grid = MeshGrid()

    def construct(self, cam2world_matrix, intrinsics, resolutions, anchors, region_size):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolutions: (N, 1)
        anchors: (N, 2)
        region_size: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """

        N, M = cam2world_matrix.shape[0], region_size**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = self.mesh_grid(region_size, region_size)
        uv = uv.flip(dims=(0,)).reshape(2, -1).swapaxes(1, 0)
        uv = uv.unsqueeze(0).tile((cam2world_matrix.shape[0], 1, 1))

        # anchors are indexed as normal (row, col) but uv is indexed as (x, y)
        x_cam = (uv[:, :, 0].view((N, -1)) + anchors[:, 1].unsqueeze(-1)) * (1.0 / resolutions) + (0.5 / resolutions)
        y_cam = (uv[:, :, 1].view((N, -1)) + anchors[:, 0].unsqueeze(-1)) * (1.0 / resolutions) + (0.5 / resolutions)
        z_cam = mint.ones((N, M))

        x_lift = (
            (
                x_cam
                - cx.unsqueeze(-1)
                + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
                - sk.unsqueeze(-1) * y_cam / fy.unsqueeze(-1)
            )
            / fx.unsqueeze(-1)
            * z_cam
        )
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = mint.stack((x_lift, y_lift, z_cam, mint.ones_like(z_cam, dtype=ms.float32)), dim=-1)

        _opencv2blender = (
            ms.Tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=ms.float32,
            )
            .unsqueeze(0)
            .tile((N, 1, 1))
        )

        cam2world_matrix = mint.bmm(cam2world_matrix, _opencv2blender)

        world_rel_points = mint.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = ray_dirs / ops.norm(ray_dirs, dim=2, keepdim=True)

        ray_origins = cam_locs_world.unsqueeze(1).tile((1, ray_dirs.shape[1], 1))

        return ray_origins, ray_dirs
