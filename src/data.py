"""Dataset loading utilities for the phishing detection project."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data") / "5.urldata.csv"
TARGET_COLUMN = "Label"
DOMAIN_COLUMN = "Domain"


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Load the phishing detection dataset from *path*.

    Raises ``FileNotFoundError`` with copy instructions if the file is missing.

    Parameters
    ----------
    path:
        Location of the CSV file.  Defaults to ``data/5.urldata.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataset including the ``Domain`` column and the ``Label`` target.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n\n"
            "To set up the dataset:\n"
            "  1. Clone https://github.com/gangeshbaskerr/Phishing-Website-Detection\n"
            "  2. Copy DataFiles/5.urldata.csv into this project as data/5.urldata.csv\n"
            "  3. Re-run the notebook from the root of this repository."
        )
    return pd.read_csv(path)


def get_feature_matrix(df: pd.DataFrame):
    """Split *df* into the feature matrix X and target vector y.

    The ``Domain`` and ``Label`` columns are excluded from X because:
    - ``Domain`` is a raw text identifier that would introduce meaningless
      ordering if treated as a numeric feature.
    - ``Label`` is the supervised target variable.

    Parameters
    ----------
    df:
        Full dataset as returned by :func:`load_dataset`.

    Returns
    -------
    X : pd.DataFrame
        Numeric engineered feature columns only.
    y : pd.Series
        Binary target (0 = legitimate, 1 = phishing).
    """
    excluded = [DOMAIN_COLUMN, TARGET_COLUMN]
    X = df.drop(columns=excluded)
    y = df[TARGET_COLUMN]
    return X, y
