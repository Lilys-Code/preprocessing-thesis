import json
import os
import random
import numpy as np
import tensorflow as tf

from src.models import build_resnet_model, build_mobilenet_model, build_efficientnet_model
from src.training import train_model
from src.evaluation import evaluate_model
from src.utils import (
    time_function, 
    save_json, 
    save_csv, 
    timestamp, 
    get_data_generators, 
    compute_class_weights, 
    precompute_all,
)
from src.preprocessing import (
    clahe_pipeline,
    hsv_pipeline,
    median_mean_hybrid,
    histogram_eq_pipeline,
    sharpen_pipeline,
    leaf_segment_pipeline,
)

PROGRESS_FILE = "logs/progress.json"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)


def load_progress():
    """Load the results and completion state from a previous run, if one exists.

    Returns a tuple of (completed, results, histories). If no progress file is
    found, or if the file is corrupt, empty defaults are returned so the run
    starts from scratch.
    """
    if not os.path.exists(PROGRESS_FILE):
        return set(), [], {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        completed = set(data.get("completed", []))
        results = data.get("results", [])
        histories = data.get("histories", {})
        return completed, results, histories
    except (json.JSONDecodeError, KeyError):
        print("[WARNING] Progress file is corrupt — starting from scratch.")
        return set(), [], {}


def save_progress(completed, results, histories):
    """Write the current experiment state to disk so the run can be resumed.

    The progress file stores the set of completed experiment keys, accumulated
    results, and full training histories. This allows long-running experiment
    grids to be interrupted and continued without redoing finished combinations.
    """
    os.makedirs("logs", exist_ok=True)
    data = {
        "completed": list(completed),
        "results": results,
        "histories": histories,
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _merge_histories(h1, h2):
    """Concatenate the loss and metric curves from two Keras History objects.

    Used to combine the initial head-only training history with the fine-tuning
    history into a single dict so both phases can be plotted together.
    """
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + (h2.history[key] if h2 is not None else [])
    return merged

def run():
    data_dir = "data/raw"
    preprocessed_dir = "data/preprocessed"

    pipelines = {
        "baseline": None,
        "clahe": clahe_pipeline,
        "hsv": hsv_pipeline,
        "median_mean_hybrid": median_mean_hybrid,
        "histogram_eq": histogram_eq_pipeline,
        "sharpen": sharpen_pipeline,
        "leaf_segment": leaf_segment_pipeline,
    }

    models_config = {
        "resnet50": build_resnet_model,
        "mobilenetV2": build_mobilenet_model,
        "efficientnet_b0": build_efficientnet_model,
    }

    # Build the dict of non-baseline pipelines to pass to precompute_all.
    # The baseline uses raw images, so it is excluded from pre-processing.
    non_baseline_pipelines = {k: v for k, v in pipelines.items() if v is not None}
    print("\n" + "="*60)
    print("Precomputing pipeline outputs to disk ...")
    print("="*60)
    precompute_all(data_dir, preprocessed_dir, non_baseline_pipelines, img_size=(224, 224))
    print("Precompute step complete.")

    completed, results, histories = load_progress()
    total = len(models_config) * len(pipelines)
    print(f"\nResuming: {len(completed)}/{total} experiments already completed.")
    if completed:
        print("Completed so far:", ", ".join(sorted(completed)))

    for model_name, model_builder in models_config.items():
        print(f"\n{'='*60}")
        print(f"Building experiments with model: {model_name.upper()}")
        print(f"{'='*60}")

        for pipeline_name, pipeline in pipelines.items():
            exp_key = f"{model_name}_{pipeline_name}"
            print(f"\n=== Running experiment: {model_name} + {pipeline_name} ===")

            if exp_key in completed:
                print(f"    [SKIP] Already completed — skipping.")
                continue

            if pipeline is None:
                # Baseline: load raw images with standard rescaling
                pipeline_dir = None
            else:
                # Non-baseline: load from the precomputed directory; no runtime
                # pipeline function needed because images are already processed.
                pipeline_dir = os.path.join(preprocessed_dir, pipeline_name)

            (train_gen, val_gen, test_gen), prep_train_time = time_function(
                get_data_generators,
                data_dir,
                img_size=(224, 224),
                batch_size=32,
                seed=SEED,
                preprocessed_dir=pipeline_dir,
            )

            class_weight = compute_class_weights(train_gen)

            num_classes = len(train_gen.class_indices)
            model, base_model = model_builder((224, 224, 3), num_classes)

            checkpoint_dir = f"logs/checkpoints/{model_name}_{pipeline_name}"
            (history, history_fine), train_time = time_function(
                train_model,
                model,
                train_gen,
                val_gen,
                epochs=10,
                base_model=base_model,
                fine_tune_layers=20,
                fine_tune_epochs=10,
                class_weight=class_weight,
                checkpoint_dir=checkpoint_dir,
            )

            # Evaluate on the held-out test set, which was not used during training or selection
            class_names = sorted(test_gen.class_indices, key=test_gen.class_indices.get)
            metrics, report = evaluate_model(model, test_gen, class_names=class_names)

            histories[exp_key] = _merge_histories(history, history_fine)

            results.append({
                "model": model_name,
                "experiment": pipeline_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "prep_train_time_sec": prep_train_time,
                "train_time_sec": train_time,
                "inference_time_sec": metrics["inference_time_sec"],
                "per_class_f1": metrics["per_class_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            })

            print(report)

            completed.add(exp_key)
            save_progress(completed, results, histories)
            print(f"    [PROGRESS] Saved ({len(completed)}/{total} done).")

            del model, base_model, train_gen, val_gen, test_gen
            tf.keras.backend.clear_session()

    os.makedirs("logs", exist_ok=True)
    ts = timestamp()

    save_json(results, f"logs/{ts}/results_{ts}.json")
    save_csv(results, f"logs/{ts}/results_{ts}.csv")
    save_json(histories, f"logs/{ts}/histories_{ts}.json")

    print("\n" + "="*60)
    print("=== FINAL SUMMARY ===")
    print("="*60)
    print(f"(Delete {PROGRESS_FILE} to start from scratch on the next run.")
    for r in results:
        print(
            f"Model: {r['model']:12} | Pipeline: {r['experiment']:20} "
            f"| Accuracy: {r['accuracy']:.4f} | F1: {r['f1_score']:.4f}"
        )


if __name__ == "__main__":
    run()
