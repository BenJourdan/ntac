"""Download and unzip the FlyWire data if not already cached."""

import os
import urllib.request
import zipfile

DATA_URL = (
    "https://github.com/BenJourdan/ntac/releases/download/v0.1.0/dynamic_data.zip"
)
def download_flywire_data(
    data_url=DATA_URL,
    cache_dir=os.path.join(os.path.expanduser("~"), ".ntac"),
    zip_path=None,
    data_dir=None,
    verbose=False):
    """Download and unzip the FlyWire data if not already cached.

    Parameters
    ----------
    data_url : str
        URL to download the FlyWire data from.
    cache_dir : str
        Directory to cache the downloaded data.
    zip_path : str
        Path to the zip file.
    data_dir : str
        Directory to extract the data to.
    verbose : bool
        If True, print progress messages.

    """
    if zip_path is None:
        zip_path = os.path.join(cache_dir, "dynamic_data.zip")
    if data_dir is None:
        data_dir = os.path.join(cache_dir, "flywire_data")

    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Download if missing
    if not os.path.exists(zip_path):
        if verbose:
            print("Downloading FlyWire data...")
        urllib.request.urlretrieve(data_url, zip_path)
        if verbose:
            print("Download complete.")

    # Step 2: Unzip if not already done
    if not os.path.exists(data_dir):
        if verbose:
            print("Unzipping FlyWire data...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        if verbose:
            print("Unzip complete.")

    if verbose:
        print(f"FlyWire data available at: {data_dir}")
    return data_dir
