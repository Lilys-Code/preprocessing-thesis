from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models


def build_efficientnet_model(input_shape, num_classes):
    """Build an EfficientNetB0-based classifier with a custom classification head.

    EfficientNetB0 is loaded with ImageNet weights and all base layers are frozen
    for the initial training phase. The head follows the same structure used across
    all models in this project: global average pooling, a dense layer with dropout,
    and a softmax output sized to the number of classes.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base during initial training so only the new head learns
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model
