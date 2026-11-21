import tensorflow as tf
from tensorflow.keras import layers, models


def build_sender_model(train_ds: tf.data.Dataset) -> tf.keras.Model:
    #sets the settings for text vectoring
    vec = layers.TextVectorization(
        max_tokens=100,
        #preserves special characters but gets rid of case
        standardize="lower",
        #splits character, this is supposedly important?
        split="character",
        #output is ints
        output_mode="int",
        output_sequence_length=80,
    )
    #builds the data table for the model
    vec.adapt(train_ds.map(lambda x, _: x["sender"]).prefetch(tf.data.AUTOTUNE))

    #looks at sections labeled sender which was set in our data setup method
    inputs = layers.Input(shape=(1,), dtype=tf.string, name="sender")

    #sets up the data
    x = vec(inputs)
    x = layers.Embedding(100, 32)(x)
    x = layers.Conv1D(64, kernel_size=4, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation="relu", name="sender_feat")(x)

    # output is a probability from 0.0 - 1.0
    outputs = layers.Dense(1, activation="sigmoid", name="sender_prob")(x)

    #compiles the results to form the model
    model = models.Model(inputs, outputs, name="sender_model")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def build_text_model(train_ds: tf.data.Dataset, part_name: str, max_tokens: int, seq_len: int) -> tf.keras.Model:
    #sets the settings for text vectoring
    vec = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
    )
    #builds the data table for the model
    vec.adapt(train_ds.map(lambda x, _: x[part_name]).prefetch(tf.data.AUTOTUNE))

    #looks at sections labeled sender which was set in our data setup method
    inputs = layers.Input(shape=(1,), dtype=tf.string, name=part_name)

    #sets up the data
    x = vec(inputs)
    x = layers.Embedding(max_tokens, 128)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3, name=f"{part_name}_feat")(x)
    # output is a probability from 0.0 - 1.0
    outputs = layers.Dense(1, activation="sigmoid", name=f"{part_name}_prob")(x)
    #compiles the results to form the model
    model = models.Model(inputs, outputs, name=f"{part_name}_model")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def build_url_model(train_ds: tf.data.Dataset | None = None) -> tf.keras.Model:
    inputs = layers.Input(shape=(1,), dtype=tf.float32, name="urls")
    outputs = layers.Dense(1, activation="sigmoid", name="urls_prob")(inputs)

    model = models.Model(inputs, outputs, name="urls_model")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model