import os
import cv2
import numpy as np


def precompute_all(data_dir, output_dir, pipelines, img_size=(224, 224)):
    """Pre-process every image in `data_dir` through each pipeline in `pipelines`."""
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )

    for pipeline_name, pipeline_fn in pipelines.items():
        print(f"\n[precompute] Pipeline: {pipeline_name}")
        total_written = 0
        total_skipped = 0

        for class_name in class_names:
            src_class_dir = os.path.join(data_dir, class_name)
            dst_class_dir = os.path.join(output_dir, pipeline_name, class_name)
            os.makedirs(dst_class_dir, exist_ok=True)

            for fname in os.listdir(src_class_dir):
                src_path = os.path.join(src_class_dir, fname)
                if not os.path.isfile(src_path):
                    continue

                # Preserve the original stem but always write as .jpg
                stem = os.path.splitext(fname)[0]
                dst_path = os.path.join(dst_class_dir, stem + ".jpg")

                if os.path.exists(dst_path):
                    total_skipped += 1
                    continue

                img_bgr = cv2.imread(src_path)
                if img_bgr is None:
                    print(f"  [WARN] Could not read {src_path} — skipping.")
                    continue

                img_bgr = cv2.resize(img_bgr, (img_size[1], img_size[0]))

                # Pipelines expect a batch of float32 images in [0, 1]
                img_float = img_bgr.astype(np.float32) / 255.0
                batch = np.expand_dims(img_float, axis=0)
                result = pipeline_fn(batch)[0]

                # Convert back to uint8 BGR for JPEG encoding
                result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)

                # HSV pipeline outputs HSV channels, not BGR — convert back so
                # ImageDataGenerator can load the file as a normal RGB image.
                # All other pipelines already output BGR-ordered float arrays.
                if pipeline_name == "hsv":
                    result_u8_h = (result_u8[:, :, 0] * 180 / 255).astype(np.uint8)
                    result_u8_sv = result_u8[:, :, 1:]
                    hsv_u8 = np.concatenate(
                        [result_u8_h[:, :, np.newaxis], result_u8_sv], axis=2
                    )
                    result_u8 = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)

                cv2.imwrite(dst_path, result_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])
                total_written += 1

        print(f"  Done — {total_written} written, {total_skipped} already existed.")
