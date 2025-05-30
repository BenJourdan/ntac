"""ntac package initialization."""

from .visualizer import Visualizer
from .data import download_flywire_data


from . import seeded
from . import unseeded

__all__ = [
    "Visualizer",
    "download_flywire_data",
    "seeded",
    "unseeded",
]

def main() -> None:
    """Run the main entry point of the ntac package."""
    print("Hello from ntac!")
    download_flywire_data(verbose=True)