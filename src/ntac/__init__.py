"""ntac package initialization."""
import numpy as np


from .visualizer import Visualizer
from .data import download_flywire_data
from scipy.sparse import csr_array



from .graph_data import GraphData, FAFBData
from .seeded import Ntac

def sbm(n, k, p_in_range=(0.1, 0.9), p_out_range=(0.1, 0.9), seed=None):
    """
    Generate a synthetic undirected graph using the Stochastic Block Model (SBM).

    This function creates an adjacency matrix and block labels using heterogeneous
    within-block (`p_in_range`) and between-block (`p_out_range`) connection probabilities.

    Parameters
    ----------
    n : int
        Total number of nodes.
    k : int
        Number of blocks (clusters).
    p_in_range : tuple of float
        Range from which to sample within-block probabilities.
    p_out_range : tuple of float
        Range from which to sample between-block probabilities.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    A : np.ndarray
        Adjacency matrix of the generated graph (symmetric, unweighted).
    labels : np.ndarray
        Array of block assignments for each node.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # determine block sizes as evenly as possible
    base_size, remainder = divmod(n, k)
    sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]
    
    # assign block labels
    labels = np.repeat(np.arange(k), sizes)
    
    # sample p_in for each block
    p_in = np.random.uniform(p_in_range[0], p_in_range[1], size=k)
    
    # sample p_out for each pair of blocks
    p_out = np.random.uniform(p_out_range[0], p_out_range[1], size=(k, k))
    np.fill_diagonal(p_out, 0.0)  # we don't use diagonal entries (they're replaced by p_in)
    
    # build full probability matrix
    P = np.zeros((n, n))
    idx = np.cumsum([0] + sizes)
    for i in range(k):
        for j in range(k):
            block = (slice(idx[i], idx[i+1]), slice(idx[j], idx[j+1]))
            if i == j:
                P[block] = p_in[i]
            else:
                P[block] = p_out[i, j]
                
    # sample upper triangle
    U = np.triu(np.random.rand(n, n) < P, 1).astype(int)
    A = U + U.T  # symmetrize
    
    return A, labels

__all__ = [
    "Visualizer",
    "download_flywire_data",
    "Ntac",
    "GraphData",
    "FAFBData",
    "sbm"
]

def main() -> None:
    """Run the main entry point of the ntac package."""
    print("Hello from ntac!")
    download_flywire_data(verbose=True)