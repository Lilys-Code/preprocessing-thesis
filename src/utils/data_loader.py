import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42), class_names

def get_data_generators(data_dir, img_size=(224, 224), batch_size=32, preprocessing_function=None, validation_split=0.2, rescale=1./255):
    datagen = ImageDataGenerator(
        rescale=rescale,
        validation_split=validation_split,
        preprocessing_function=preprocessing_function
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen
