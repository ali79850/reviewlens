"""
Download and subsample Amazon Reviews 2023 (Electronics).

Note on data access: the McAuley-Lab/Amazon-Reviews-2023 Hugging Face repo
only converted its *_meta_* configs to Parquet. Review data was never
migrated and its `datasets`-library loading script no longer runs under
`datasets>=4.0` (script-based loading was removed for security reasons).
We therefore stream the gzip-compressed JSONL directly from the dataset
authors' own host (linked from the official dataset card / GitHub) and
reservoir-sample while streaming, so memory usage stays bounded regardless
of the source file's size.

Run from project root: python scripts/download_data.py
"""
import gzip
import json
import random
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, "src")  # temporary until reviewlens is pip-installed (Phase 3)
from reviewlens.utils.paths import load_config, resolve

SOURCE_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/{domain}.jsonl.gz"
)


def download_gz(url: str, dest_path) -> None:
    """Stream-download a file with a byte-progress bar. Skips if already present."""
    if dest_path.exists():
        print(f"Found existing download at {dest_path}, skipping re-download.")
        return

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with open(tmp_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"Downloading {dest_path.name}"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))
    tmp_path.rename(dest_path)


def reservoir_sample_jsonl_gz(gz_path, target: int, min_chars: int, seed: int) -> list[dict]:
    """Read a gzipped JSONL file line-by-line and keep a uniform random sample
    of `target` valid rows, without ever loading the full file into memory."""
    random.seed(seed)
    reservoir: list[dict] = []
    seen = 0

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning + sampling reviews"):
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = (example.get("text") or "").strip()
            rating = example.get("rating")
            if len(text) < min_chars or rating is None:
                continue

            row = {
                "text": text,
                "title": example.get("title", ""),
                "rating": float(rating),
                "verified_purchase": example.get("verified_purchase"),
                "timestamp": example.get("timestamp"),
            }

            seen += 1
            if len(reservoir) < target:
                reservoir.append(row)
            else:
                j = random.randint(0, seen - 1)
                if j < target:
                    reservoir[j] = row

    return reservoir


def main():
    cfg = load_config()
    ds_cfg = cfg["dataset"]

    domain = ds_cfg["hf_config"].replace("raw_review_", "")
    url = SOURCE_URL_TEMPLATE.format(domain=domain)

    raw_dir = resolve(cfg["paths"]["data_raw"])
    gz_path = raw_dir / f"{domain}.jsonl.gz"

    print(f"Source: {url}")
    download_gz(url, gz_path)

    rows = reservoir_sample_jsonl_gz(
        gz_path,
        target=ds_cfg["target_samples"],
        min_chars=ds_cfg["min_text_chars"],
        seed=ds_cfg["shuffle_seed"],
    )

    df = pd.DataFrame(rows)
    print("Actual columns collected:", list(df.columns))

    out_path = raw_dir / "amazon_electronics_raw.parquet"
    df.to_parquet(out_path, index=False)

    # Real stats only — nothing here is estimated
    stats = {
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": url,
        "sampling_method": "reservoir sampling over full JSONL stream (uniform, unbiased by file order)",
        "n_rows": len(df),
        "columns": list(df.columns),
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
        "text_length_chars": {
            "mean": round(df["text"].str.len().mean(), 1),
            "median": float(df["text"].str.len().median()),
            "min": int(df["text"].str.len().min()),
            "max": int(df["text"].str.len().max()),
        },
        "duplicate_text_rows": int(df["text"].duplicated().sum()),
    }
    metrics_dir = resolve(cfg["paths"]["results"] + "/metrics")
    with open(metrics_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Stats written to {metrics_dir / 'dataset_stats.json'}")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()