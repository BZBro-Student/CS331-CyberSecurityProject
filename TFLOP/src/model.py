import tensorflow as tf
from tensorflow.keras import layers, models

def build_sender_model(train_ds: tf.data.Dataset) -> tf.keras.Model:
    vec = layers.TextVectorization(
        max_tokens=100,
        split="character",
        output_mode="int",
        output_sequence_length=80,
    )
    vec.adapt(train_ds.map(lambda x, _: x["sender"]))

    inputs = layers.Input(shape=(1,), dtype=tf.string, name="sender")
    x = vec(inputs)
    x = layers.Embedding(100, 32)(x)
    x = layers.Conv1D(64, kernel_size=4, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="sender_prob")(x)

    model = models.Model(inputs, outputs, name="sender_model")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_text_model(
    train_ds: tf.data.Dataset, part_name: str, max_tokens: int, seq_len: int
) -> tf.keras.Model:
    vec = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
    )
    vec.adapt(train_ds.map(lambda x, _: x[part_name]))

    inputs = layers.Input(shape=(1,), dtype=tf.string, name=part_name)
    x = vec(inputs)
    x = layers.Embedding(max_tokens, 128)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name=f"{part_name}_prob")(x)

    model = models.Model(inputs, outputs, name=f"{part_name}_model")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_url_model(train_ds: tf.data.Dataset) -> tf.keras.Model:
    norm = layers.Normalization()
    norm.adapt(train_ds.map(lambda x, _: x["urls"]))

    inputs = layers.Input(shape=(1,), dtype=tf.float32, name="urls")
    x = norm(inputs)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="urls_prob")(x)

    model = models.Model(inputs, outputs, name="urls_model")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model