import gdown
import shutil
from pathlib import Path
from pipeline.data.constants import DRIVE_IDS


def test_drive_ids():
    download_dir = Path(__file__).parent / "test_artefacts"
    download_dir.mkdir(parents=True, exist_ok=True)
    download_dir_str = f"{str(download_dir).rstrip('/')}/"

    # Test if all drive ids are valid
    for key, value in DRIVE_IDS.items():
        name = gdown.download(id=value, output=download_dir_str, quiet=True)
        assert name == str(download_dir / key), f"Downloaded file name {name} does not match expected name {key}"

    shutil.rmtree(download_dir)