import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(
    model,
    train_gen,
    val_gen,
    epochs=10,
    base_model=None,
    fine_tune_layers=20,
    fine_tune_epochs=10,
    class_weight=None,
    checkpoint_dir="logs/checkpoints",
):
    """Train a model in two phases: head-only training followed by optional fine-tuning.

    In the first phase the frozen base is kept fixed and only the new classification
    head is trained for `epochs` epochs. Early stopping monitors validation loss and
    the best checkpoint is saved to `checkpoint_dir`.

    If `base_model` is provided, a second fine-tuning phase unfreezes the top
    `fine_tune_layers` of the base and trains for a further `fine_tune_epochs` epochs
    at a lower learning rate. This allows the upper convolutional layers to adapt to
    the plant disease domain without disrupting the lower-level feature detectors.

    Class weights are passed through to both phases to address any class imbalance
    in the training set.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    history_fine = None
    if base_model is not None:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs,
            class_weight=class_weight,
            callbacks=callbacks,
        )

    return history, history_fine
