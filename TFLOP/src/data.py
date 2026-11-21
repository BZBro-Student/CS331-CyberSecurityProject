from pathlib import Path
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
import config

def load_data():
    dataset_dir = Path(kagglehub.dataset_download(config.DATA))
    csv_path = dataset_dir / "CEAS_08.csv"
    df = pd.read_csv(csv_path)

    features = {
            "sender": df["sender"].fillna("").astype(str).values.reshape(-1, 1),
            "subject": df["subject"].fillna("").astype(str).values.reshape(-1, 1),
            "body": df["body"].fillna("").astype(str).values.reshape(-1, 1),
            "urls": pd.to_numeric(df["urls"], errors="coerce")
            .fillna(0)
            .astype("float32")
            .values.reshape(-1, 1),
        }
    labels = df["label"].astype("float32").values

    total = len(df)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)


    full_ds = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(
        buffer_size=total, seed=42, reshuffle_each_iteration=False
    )
    train_ds = (
        full_ds.take(train_size)
        .shuffle(10_000, seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        full_ds.skip(train_size)
        .take(val_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        full_ds.skip(train_size + val_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds