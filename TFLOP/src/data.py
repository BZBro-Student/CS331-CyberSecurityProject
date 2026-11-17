from pathlib import Path
import kagglehub
import tensorflow as tf

def load_data():
    dataset_dir = Path(kagglehub.dataset_download(config.DATASET_SLUG))
    csv_path = dataset_dir / "CEAS_08.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CEAS_08.csv not found in {dataset_dir}")