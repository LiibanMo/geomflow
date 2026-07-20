from typing import Union, List
import numpy as np
import torch

ArrayLike = Union[
    np.ndarray,
    List[float],
    List[List[float]],
    torch.Tensor,
    "np.typing.NDArray[np.floating]",
]


def preprocess(
    data: ArrayLike,
    *,
    normalize: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert ArrayLike input to a torch.Tensor ready for CNF training.

    Args:
        data: Input in any of: numpy array, list, list-of-lists, or torch tensor.
        normalize: If True, standardize to zero mean / unit variance.
        dtype: Target torch dtype (default float32).

    Returns:
        Torch tensor of shape (N, D) where N = number of samples,
        D = manifold dimension. Gradient tracking enabled.
    """
    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype=dtype)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype=dtype)
    elif isinstance(data, list):
        arr = np.asarray(data, dtype=np.float32)
        tensor = torch.from_numpy(arr).to(dtype=dtype)
    else:
        raise TypeError(
            f"Unsupported type: {type(data)}. "
            f"Expected numpy array, list, or torch.Tensor."
        )

    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    if normalize:
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True).clamp(min=1e-8)
        tensor = (tensor - mean) / std

    tensor = tensor.requires_grad_(True)
    return tensor

