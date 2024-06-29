from jaxtyping import Float
from torch import Tensor
import torch
from src.geometry import homogenize_points, transform_world2cam, project
from src.provided_code import plot_point_cloud

def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    canvas = torch.ones((extrinsics.shape[0], *resolution))

    xyz = homogenize_points(vertices)
    xyz = xyz.unsqueeze(0)
    extrinsics = extrinsics.unsqueeze(1)
    intrinsics = intrinsics.unsqueeze(1)

    xyz = transform_world2cam(xyz, extrinsics)
    xy = project(xyz, intrinsics)

    for i, vertices in enumerate(xy):
        v = vertices.clone()
        v[:, 0] *= resolution[0]
        v[:, 1] *= resolution[1]
        v = torch.round(v).long()
        v = v[(v[:, 0] >= 0) * (v[:, 0] < resolution[0]) * (v[:, 1] >= 0) * (v[:, 1] < resolution[1])]
        ind = v[:, 1] * resolution[0] + v[:, 0]
        canvas[i] = canvas[i].reshape(-1).index_fill_(0, ind, 0).reshape(resolution)

    return canvas
