"""ntac package initialization."""

from .data import GraphData
from .ntac import Ntac

__all__ = ["GraphData", "Ntac"]


def main() -> None:
    """Run the main entry point of the ntac package."""
    print("Hello from ntac!")
