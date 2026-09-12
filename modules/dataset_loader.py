

"""Dataset Loader Module for CICIDS2017.

Handles reading raw CICIDS2017 network traffic flow CSVs and sanitizing column headers.
"""

from pathlib import Path
from typing import Union
import pandas as pd


def load_raw_data(path: Union[str, Path]) -> pd.DataFrame:
    """Reads a CICIDS2017 CSV file into a DataFrame, strips whitespace from column names, and returns it.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the raw CICIDS2017 CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the raw data with whitespace stripped from all column headers.
    """
    path_str = str(path)
    try:
        df = pd.read_csv(path_str)
    except UnicodeDecodeError:
        df = pd.read_csv(path_str, encoding="latin-1")

    # Strip leading and trailing whitespace from column names
    df.columns = df.columns.str.strip()
    return df
