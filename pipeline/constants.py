from pathlib import Path


SEED = 42

ARTEFACTS_DIR = Path(__file__).parent.parent / "artefacts"

if not ARTEFACTS_DIR.exists():
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)