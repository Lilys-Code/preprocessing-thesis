import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(
    data_dir,
    img_size=(224, 224),
    batch_size=32,
    train_split=0.70,
    val_split=0.15,
    rescale=1.0 / 255,
    seed=42,
    preprocessed_dir=None,
):
    """Build train, validation, and test generators from a directory of class folders.

    The dataset is split into train/val/test using a stratified 70/15/15 ratio so
    that each subset preserves the original class distribution. The training
    generator applies random augmentation (rotations, flips, zoom, brightness
    jitter) to improve generalisation. Validation and test generators use only
    rescale so that evaluation is deterministic.
    """
    
    # Determine the root from which to enumerate files. When a preprocessed
    # directory exists, use it and disable the runtime pipeline function.
    if preprocessed_dir is not None and os.path.isdir(preprocessed_dir):
        image_root = preprocessed_dir
        rescale = 1.0 / 255
    else:
        image_root = data_dir

    filepaths, labels = [], []
    class_names = sorted(
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d))
    )

    for class_name in class_names:
        class_dir = os.path.join(image_root, class_name)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath):
                filepaths.append(fpath)
                labels.append(class_name)

    # Stratified 3-way split — train first, then divide the remainder into val and test
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=1.0 - train_split, stratify=labels, random_state=seed
    )
    # Proportion of the held-out portion that becomes the test set
    test_ratio = (1.0 - train_split - val_split) / (1.0 - train_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, stratify=y_temp, random_state=seed
    )

    train_df = pd.DataFrame({"filepath": X_train, "label": y_train})
    val_df = pd.DataFrame({"filepath": X_val, "label": y_val})
    test_df = pd.DataFrame({"filepath": X_test, "label": y_test})

    # Augmentation is applied only during training; val/test use a fixed pipeline
    train_datagen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
    )
    eval_datagen = ImageDataGenerator(
        rescale=rescale,
    )

    common = dict(
        x_col="filepath",
        y_col="label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
    )

    train_gen = train_datagen.flow_from_dataframe(
        train_df, shuffle=True, seed=seed, **common
    )
    val_gen = eval_datagen.flow_from_dataframe(
        val_df, shuffle=False, seed=seed, **common
    )
    test_gen = eval_datagen.flow_from_dataframe(
        test_df, shuffle=False, seed=seed, **common
    )

    return train_gen, val_gen, test_gen


def compute_class_weights(train_gen):
    """Compute per-class weights inversely proportional to class frequency.

    Passes the training generator's class labels to sklearn's compute_class_weight
    with the 'balanced' strategy, which scales each class weight so that rarer
    classes contribute more to the loss. The result is a dict mapping class index
    to weight, ready to pass to model.fit as `class_weight`.
    """
    classes = train_gen.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes,
    )
    return dict(enumerate(class_weights))

