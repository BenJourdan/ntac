"""ntac package initialization."""

from .visualizer import Visualizer
from .data import download_flywire_data
from scipy.sparse import csr_array


# from . import seeded
# from . import unseeded

from .seeded import SeededNtac as Ntac



__all__ = [
    "Visualizer",
    "download_flywire_data",
    "Ntac"
]

def main() -> None:
    """Run the main entry point of the ntac package."""
    print("Hello from ntac!")
    download_flywire_data(verbose=True)