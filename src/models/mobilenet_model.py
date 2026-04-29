from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


def build_mobilenet_model(input_shape, num_classes):
    """Build a MobileNetV2-based classifier with a custom classification head.

    MobileNetV2 is loaded with ImageNet weights and all base layers are frozen
    for the initial training phase. Its depthwise separable convolutions make it
    considerably lighter than ResNet50, which is useful for comparing accuracy
    against computational cost. The classification head matches the other models
    in this project for consistency.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

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
