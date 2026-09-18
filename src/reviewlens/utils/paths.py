"""Project path resolution — avoids hardcoded/relative-path bugs on Windows."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    """Load config.yaml as a plain dict."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative_path: str) -> Path:
    """Turn a config-relative path (e.g. 'data/raw') into an absolute Path,
    creating parent directories if needed."""
    p = PROJECT_ROOT / relative_path
    p.mkdir(parents=True, exist_ok=True)
    return p
