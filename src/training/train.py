from tensorflow.keras.optimizers import Adam


def train_model(model, train_gen, val_gen, epochs=10, base_model=None, fine_tune_layers=20, fine_tune_epochs=10):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    if base_model is not None:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs
        )
        return history_fine

    return history