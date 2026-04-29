from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models


def build_resnet_model(input_shape, num_classes):
    """Build a ResNet50-based classifier with a custom classification head.

    The ResNet50 convolutional base is loaded with ImageNet weights and its layers
    are initially frozen so that only the new head is trained first. After this
    initial phase, the caller can unfreeze the top layers for fine-tuning.
    The head consists of global average pooling, a dense layer with dropout for
    regularisation, and a softmax output layer sized to the number of classes.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

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