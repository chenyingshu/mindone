from typing import Callable, Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

import mindone.models.threestudio as threestudio
from mindone.models.threestudio.models.mesh import Mesh


class IsosurfaceHelper(nn.Cell):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> Tensor:
        raise NotImplementedError

    def construct(self):
        raise NotImplementedError


class MarchingCubeCPUHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        import mcubes

        self.mc_func: Callable = mcubes.marching_cubes
        self._grid_vertices: Optional[Tensor] = None
        self._dummy: Tensor
        self.register_buffer("_dummy", mint.zeros(0, dtype=ms.float32), persistent=False)

    @property
    def grid_vertices(self) -> Tensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                mint.linspace(*self.points_range, self.resolution),
                mint.linspace(*self.points_range, self.resolution),
                mint.linspace(*self.points_range, self.resolution),
            )
            x, y, z = ops.meshgrid(x, y, z, indexing="ij")
            verts = mint.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def construct(
        self,
        level: Tensor,
        deformation: Optional[Tensor] = None,
    ) -> Mesh:
        if deformation is not None:
            threestudio.info(f"{self.__class__.__name__} does not support deformation. Ignoring.")
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(level.asnumpy(), 0.0)  # transform to numpy
        v_pos, t_pos_idx = (
            Tensor(v_pos).float(),
            Tensor(t_pos_idx.astype(np.int64)).long(),
        )  # transform back to ms tensor on npu
        v_pos = v_pos / (self.resolution - 1.0)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)


class MarchingTetrahedraHelper(IsosurfaceHelper):
    def __init__(self, resolution: int, tets_path: str):
        super().__init__()
        self.resolution = resolution
        self.tets_path = tets_path

        self.triangle_table: Tensor
        self.register_buffer(
            "triangle_table",
            Tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [1, 0, 2, -1, -1, -1],
                    [4, 0, 3, -1, -1, -1],
                    [1, 4, 2, 1, 3, 4],
                    [3, 1, 5, -1, -1, -1],
                    [2, 3, 0, 2, 5, 3],
                    [1, 4, 0, 1, 5, 4],
                    [4, 2, 5, -1, -1, -1],
                    [4, 5, 2, -1, -1, -1],
                    [4, 1, 0, 4, 5, 1],
                    [3, 2, 0, 3, 5, 2],
                    [1, 3, 5, -1, -1, -1],
                    [4, 1, 2, 4, 3, 1],
                    [3, 0, 4, -1, -1, -1],
                    [2, 0, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1],
                ],
                dtype=mint.long,
            ),
            persistent=False,
        )
        self.num_triangles_table: Tensor
        self.register_buffer(
            "num_triangles_table",
            Tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=mint.long),
            persistent=False,
        )
        self.base_tet_edges: Tensor
        self.register_buffer(
            "base_tet_edges",
            Tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=mint.long),
            persistent=False,
        )

        tets = np.load(self.tets_path)
        self._grid_vertices: Tensor
        self.register_buffer(
            "_grid_vertices",
            mint.from_numpy(tets["vertices"]).float(),
            persistent=False,
        )
        self.indices: Tensor
        self.register_buffer("indices", mint.from_numpy(tets["indices"]).long(), persistent=False)

        self._all_edges: Optional[Tensor] = None

    def normalize_grid_deformation(self, grid_vertex_offsets: Tensor) -> Tensor:
        return (
            (self.points_range[1] - self.points_range[0])
            / (self.resolution)  # half tet size is approximately 1 / self.resolution
            * mint.tanh(grid_vertex_offsets)
        )  # FIXME: hard-coded activation

    @property
    def grid_vertices(self) -> Tensor:
        return self._grid_vertices

    @property
    def all_edges(self) -> Tensor:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            edges = mint.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                dtype=mint.long,
                device=self.indices.device,
            )
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = mint.sort(_all_edges, dim=1)[0]
            _all_edges = mint.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        with ms._no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = mint.gather(input=edges_ex2, index=order, dim=1)
            b = mint.gather(input=edges_ex2, index=1 - order, dim=1)

        return mint.stack([a, b], -1)

    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        with ms._no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = mint.sum(occ_fx4, -1)
            valid_tets = mint.logical_and(occ_sum > 0, occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = mint.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = mint.ones((unique_edges.shape[0]), dtype=mint.long, device=pos_nx3.device) * -1
            mapping[mask_edges] = mint.arange(mask_edges.sum(), dtype=mint.long, device=pos_nx3.device)
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = mint.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = mint.pow(2, mint.arange(4, dtype=mint.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = mint.cat(
            (
                mint.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                mint.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

    def construct(
        self,
        level: Tensor,
        deformation: Optional[Tensor] = None,
    ) -> Mesh:
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(deformation)
        else:
            grid_vertices = self.grid_vertices

        v_pos, t_pos_idx = self._forward(grid_vertices, level, self.indices)

        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_level=level,
            grid_deformation=deformation,
        )

        return mesh
