from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
from torch import Tensor

import json
from PIL import Image
import numpy as np
import torch

from src.geometry import homogenize_vectors

class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""
    metadata = (path / "metadata.json" ).open('r')
    metadata = json.loads('\n'.join(metadata.readlines()))

    image_names = path.glob("images/*.png")
    images = []
    for image_name in image_names: 
        images.append(np.asarray(Image.open(image_name)))

    return PuzzleDataset(extrinsics = Tensor(metadata['extrinsics']),
                        intrinsics = Tensor(metadata['intrinsics']),
                        images = Tensor(np.array(images)))


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """
    #dataset['extrinsics'] = torch.linalg.inv(dataset['extrinsics'])

    # [R|T]
    # [Rt|RT]

    # if c2w
    # RT = [? ? -1]
    
    # if w2c
    # T = [0 0 1]

    ext = dataset['extrinsics']

    
    R = ext[..., :3, :3]
    T = ext[..., :3, -1:]
    RT = torch.matmul(R, T)

    find_z = torch.cat([abs(T), abs(RT)], dim=-2)
    z = torch.argmax(find_z, dim=-2)[0]

    if (z < 3 and T[0, z] < 0) or (z >= 3 and RT[0, z] > 0):
       T *= -1

    # TODO: automatically determine perm
    perm = [2,1,0]
    R = R[..., perm,:]
    T = T[..., perm,:]
    
    # TODO: automatically determine axis
    R[..., 0, :] *= -1
    R[..., 1, :] *= -1
    
    if z < 3:
        R = R.transpose(-1,-2)
        T = -torch.matmul(R, T)

    ext[..., :3, :3] = R
    ext[..., :3, -1:] = T
    dataset['extrinsics'] = ext

    return dataset


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    return "w2c"


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    return "x"


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    return "-y"


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    return "-z"


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    return """
            1. w2c/c2w : I looked at T and R*T. If abs(T) has the highest value, then the format is in w2c. If not, c2w
            2. the index of the look vector is determined by argmax(abs(T)) if w2c, or argmax(abs(RT)) if c2w.
            3. I brute-forced the axis rotation & signs by comparing the output images with GT.
           """
