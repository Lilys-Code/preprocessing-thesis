import cv2
import numpy as np


def _to_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.max() > 1.0:
        return np.clip(img, 0, 255).astype(np.uint8)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalize(images):
    return images / 255.0


def clahe_pipeline(images):
    processed = []
    for img in images:
        img_bgr = _to_uint8(img)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0)
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        img_out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        processed.append(img_out.astype(np.float32) / 255.0)

    return np.array(processed)


def hsv_pipeline(images):
    processed = []
    for img in images:
        img_bgr = _to_uint8(img)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hsv_float = hsv.astype(np.float32)
        hsv_float[:, :, 0] /= 180.0
        hsv_float[:, :, 1] /= 255.0
        hsv_float[:, :, 2] /= 255.0
        processed.append(hsv_float)
    return np.array(processed)


def median_mean_hybrid(images, window_size=(3, 3)):
    if isinstance(images, np.ndarray):
        imgs = images
    else:
        imgs = np.array(images)

    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=0)

    out_images = []

    for img in imgs:
        img_u8 = _to_uint8(img)

        if img_u8.ndim == 2:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        channels = cv2.split(img_u8)
        filtered_channels = []
        for ch in channels:
            median_filtered = cv2.medianBlur(ch, window_size[0])
            mean_filtered = cv2.blur(ch, window_size)
            hybrid = 0.5 * (mean_filtered.astype(np.float32) + median_filtered.astype(np.float32))
            filtered_channels.append(hybrid)

        filtered = cv2.merge(filtered_channels).astype(np.float32) / 255.0
        out_images.append(filtered)

    if isinstance(images, np.ndarray) and images.ndim == 3:
        return out_images[0]
    return np.stack(out_images, axis=0)

