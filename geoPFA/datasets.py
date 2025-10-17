"""Functions to fetch sample datasets for geoPFA."""

import pandas as pd
import pooch
from pooch.processors import Unzip
from pathlib import Path

dogbert = pooch.create(
    path=pooch.os_cache("geoPFA"),
    base_url="https://github.com/NREL/geoPFA/releases/download/{version}/",
    version="v0.0.5",
    registry={
        "heat.zip": "sha256:fc7abec6d035f7be6e31b6071ee016afa629fcb1764dc506156cc535909d9055",
        "insulation.zip": "sha256:2b16eacf32be347cf0767ee3a450eed76a92b2a654e5d249ff2fef09fab381c0",
        "producibility.zip": "sha256:5d72eb75815f86ddbb4e19bea5664c9deb27d308debb40c92068a45bbb3a94ca",
    },
)


def _get_dataset(filename: str, dataset: str) -> pd.DataFrame:
    """Fetch dataset from zip file.

    Parameters
    ----------
    filename : str
        Name of the zip file to fetch.
    dataset : str
        Name of the dataset to fetch.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested dataset.
    """
    fnames = dogbert.fetch(filename, processor=Unzip())
    inventory = {Path(f).stem: f for f in fnames}

    try:
        data = pd.read_parquet(inventory[dataset])
    except KeyError:
        source = dogbert.registry[filename]
        raise KeyError(
            f"Dataset {dataset} not included in {dogbert.get_url(filename)}."
        )

    return data


def fetch_heat(dataset: str) -> pd.DataFrame:
    """Fetch heat sample dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset to fetch.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested heat dataset.

    Examples
    --------
    >>> from geoPFA.datasets import fetch_heat
    >>> temperature = fetch_heat("temperature_model_500m")
    """
    return _get_dataset("heat.zip", dataset)


def fetch_insulation(dataset: str) -> pd.DataFrame:
    """Fetch insulation sample dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset to fetch.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested insulation dataset.

    Examples
    --------
    >>> from geoPFA.datasets import fetch_insulation
    >>> velocity = fetch_insulation("velocity_model_vp")
    """
    return _get_dataset("insulation.zip", dataset)


def fetch_producibility(dataset: str) -> pd.DataFrame:
    """Fetch producibility sample dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset to fetch.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested producibility dataset.

    Examples
    --------
    >>> from geoPFA.datasets import fetch_producibility
    >>> geology = fetch_producibility("geology")
    """
    return _get_dataset("producibility.zip", dataset)
