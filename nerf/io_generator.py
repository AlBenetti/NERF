"""Module to generate a long """
import os
from typing import Tuple, Optional
import argparse

import torch

def generate_matrix_2_matrix_data(
    shape: Tuple[int, int],
    n_obs: int = 1000,
    eps: Optional[float] = 0.001,
):
    """
    This function will generate a matrix to matrix data set.

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the data set
    eps : float, optional
        The noise to add to the data, by default 0.001
    n_obs : int
        The number of observations to generate, by default 1000

    Returns
    -------
    torch.utils.data.Dataset
        The generated data set
    """
    x = torch.rand(size=(n_obs, *shape)).view(n_obs, -1, shape[0], shape[1])
    y = (torch.sin(x) + eps * torch.randn_like(x)).view(n_obs, -1, shape[0], shape[1])

    # Create a dataset
    dataset = torch.utils.data.TensorDataset(x, y)

    return dataset

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", type=int, nargs=2, default=(3, 3))
    parser.add_argument("--eps", type=float, default=0.00001)
    parser.add_argument("--n_obs", type=int, default=1000)

    args = parser.parse_args()

    # Set the seed
    SEED = args.seed
    torch.manual_seed(SEED)

    # Generate the data set
    for fold_type in ["train", "valid"]:
        dummy_dataset = generate_matrix_2_matrix_data(
            shape=args.shape,
            eps=args.eps,
            n_obs=args.n_obs
        )

        # Save the data set relative to the current file
        path = os.path.join(os.path.dirname(__file__), "data", f"matrix_2_matrix_{fold_type}.pt")
        torch.save(dummy_dataset, path)
