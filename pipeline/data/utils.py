import logging
from pathlib import Path
from urllib.parse import urlparse

import gdown
import jsonlines
import pandas as pd

from pipeline.data.constants import DRIVE_IDS
from pipeline.constants import ARTEFACTS_DIR


logger = logging.getLogger(__name__)


def extract_domain_name(url):
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
