from jaxtyping import Float
from torch import Tensor
import torch

def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    return torch.cat([points, torch.zeros_like(points[..., :1])], dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    return torch.matmul(transform, xyz.unsqueeze(-1)).squeeze(-1)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """
    world2cam = torch.zeros_like(cam2world)
    world2cam[..., 3, 3] = 1
    world2cam[..., :3, :3] = cam2world[..., :3, :3].transpose(-1,-2)
    world2cam[..., :3, 3:] = - torch.matmul(world2cam[..., :3, :3], cam2world[..., :3, 3:])

    return torch.matmul(world2cam, xyz.unsqueeze(-1)).squeeze(-1)


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    return torch.matmul(cam2world, xyz.unsqueeze(-1)).squeeze(-1)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    xyz = xyz[..., :-1] / xyz[..., -1:]
    xy = xyz / xyz[..., -1:]

    return (torch.matmul(intrinsics, xy.unsqueeze(-1)).squeeze(-1))[..., :-1]
