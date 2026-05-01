import cv2
import numpy as np


def _to_uint8(img):
    """Convert an image array to uint8, handling both [0, 1] and [0, 255] float ranges.

    OpenCV functions generally expect uint8 input, so this helper is called at the
    start of each pipeline before any colour space conversions.
    """
    if img.dtype == np.uint8:
        return img
    if img.max() > 1.0:
        return np.clip(img, 0, 255).astype(np.uint8)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalize(images):
    """Scale pixel values from [0, 255] to [0, 1] by dividing by 255."""
    return images / 255.0


def clahe_pipeline(images):
    """Apply CLAHE to the luminance channel of each image.

    Images are converted to LAB colour space so that contrast enhancement is applied
    only to the L (lightness) channel, leaving hue and saturation unchanged. This
    improves local contrast in darker or over-exposed regions of the leaf without
    introducing colour artefacts.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = []
    for img in images:
        img_bgr = _to_uint8(img)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        img_out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        processed.append(img_out.astype(np.float32) / 255.0)

    return np.array(processed)


def hsv_pipeline(images):
    """Convert images from BGR to HSV and normalise each channel to [0, 1].

    HSV separates hue from intensity, which can make colour-based disease symptoms
    more consistent across images captured under different lighting conditions.
    Each channel is normalised to its natural OpenCV range (H: 0-180, S/V: 0-255).
    """
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
    """Apply a hybrid median-mean filter to each image channel.

    Both a median filter and a mean (box) filter are applied independently to each
    channel with the same kernel size, and their outputs are averaged. The median
    filter suppresses salt-and-pepper noise while preserving edges; the mean filter
    produces a smoother overall result. Blending the two aims to balance noise
    reduction with edge retention.
    """
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


def histogram_eq_pipeline(images):
    """Apply global histogram equalisation to the L channel in LAB colour space.

    Unlike CLAHE, this redistributes the full tonal range globally across the image
    rather than locally. It is included as a comparison point to assess whether
    adaptive or global contrast enhancement better supports disease classification.
    """
    processed = []
    for img in images:
        img_bgr = _to_uint8(img)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        merged = cv2.merge((l_eq, a, b))
        img_out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        processed.append(img_out.astype(np.float32) / 255.0)
    return np.array(processed)


def sharpen_pipeline(images):
    """Sharpen each image using a Laplacian-based unsharp masking kernel.

    The 3x3 kernel amplifies high-frequency detail such as edges and fine texture,
    which may make lesion boundaries more visually distinct. Output values are
    clipped to [0, 255] to prevent overflow artefacts from the convolution.
    """
    kernel = np.array(
        [[ 0, -1,  0],
         [-1,  5, -1],
         [ 0, -1,  0]], dtype=np.float32
    )
    processed = []
    for img in images:
        img_bgr = _to_uint8(img)
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        processed.append(np.clip(sharpened, 0, 255).astype(np.float32) / 255.0)
    return np.array(processed)


def leaf_segment_pipeline(images):
    """Remove the image background using an HSV-based colour mask.

    Each image is converted to HSV and a binary mask is created for pixels within a
    broad green/yellow-green hue range, covering typical healthy and diseased leaf
    colours. Morphological closing and opening clean up the mask by filling small
    holes and removing noise. Background pixels are then replaced with the mean
    colour of the masked foreground region.
    """
    processed = []
    for img in images:
        img_bgr = _to_uint8(img)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Hue range covering green through yellow-green leaf tones
        lower = np.array([15, 30, 30], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological close fills holes in the mask; open removes small noise blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Replace background pixels with the mean foreground colour
        mean_color = cv2.mean(img_bgr, mask=mask)[:3]
        bg = np.full(img_bgr.shape, mean_color, dtype=np.uint8)
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch > 0, img_bgr, bg)

        processed.append(result.astype(np.float32) / 255.0)
    return np.array(processed)


