import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import TYPE_CHECKING

import gdown
import jsonlines
import pandas as pd

from pipeline.data.constants import DRIVE_IDS
from pipeline.constants import ARTEFACTS_DIR


if TYPE_CHECKING:
    import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_domain_name(url: str) -> str | None:
    """
    Extract the domain name from the url.

    Args:
        url: The url to extract the domain name from.

    Returns:
        The domain name.
    """
    if not url:
        return None
    
    try:
        parsed_url = urlparse(url)
        netloc = str(parsed_url.netloc)
        return netloc.strip("www.")
    except ValueError:
        return None
    

def load_jsonl(file_path):
    lines = []
    with jsonlines.open(file_path, 'r') as f_:
        for line in f_.iter(allow_none=True):
            lines.append(line)
    return lines


def load_artefact(key):
    """Loads data."""
    path = Path(key) if Path(key).is_absolute() else Path(ARTEFACTS_DIR) / key
    if not path.exists() and not key in DRIVE_IDS:
        raise ValueError(f"Key {key} not found on disk. It also does not exist in DRIVE_IDS. Either provide a valid path or drive id."
                         f" Supported drive ids: {list(DRIVE_IDS.keys())}.")
    if not path.exists() and key in DRIVE_IDS:
        path = Path(gdown.download(id=DRIVE_IDS[key], output=f"{ARTEFACTS_DIR}/", quiet=True))

    if path.suffix == ".jsonl":
        return load_jsonl(path)
    elif path.suffix == ".tsv":
        return pd.read_csv(path, delimiter="\t")
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .tsv.")


def fix_negatives(row_idx: int, rows: list[dict], rng: np.random.Generator) -> dict:
    """
    Add the desired number of negative samples to the row.

    Args:
        row_idx: The index of the row to fix.
        rows: The rows to fix the negatives for.
        rng: The random number generator to use.

    Returns:
        The fixed row.
    """
    picked = False
    size = 7

    while not picked:
        neg_idxs = rng.choice(len(rows), size=size, replace=False)
        if row_idx not in neg_idxs:
            picked = True

    row = rows[row_idx]
    row["neg"] = [
        rows[i]["pos"][0]
        if isinstance(rows[i]["pos"], list) else rows[i]["pos"]
        for i in neg_idxs
    ]
    return row
