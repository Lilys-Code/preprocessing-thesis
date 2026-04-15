import os
import numpy as np

from src.models import build_resnet_model, build_mobilenet_model, build_efficientnet_model
from src.training import train_model
from src.evaluation import evaluate_model
from src.utils import time_function, save_json, save_csv, timestamp, get_data_generators
from src.preprocessing import (
    normalize,
    clahe_pipeline,
    hsv_pipeline,
    median_mean_hybrid,
)


def run():
    data_dir = "data/raw/"

    pipelines = {
        "baseline": None,
        "clahe": clahe_pipeline,
        "hsv": hsv_pipeline,
        "median_mean_hybrid": median_mean_hybrid,
    }

    models_config = {
        "resnet50": build_resnet_model,
        "mobilenetV2": build_mobilenet_model,
        "efficientnet_b0": build_efficientnet_model,
    }

    results = []

    for model_name, model_builder in models_config.items():
        print(f"\n{'='*60}")
        print(f"Building experiments with model: {model_name.upper()}")
        print(f"{'='*60}")

        for pipeline_name, pipeline in pipelines.items():
            print(f"\n=== Running experiment: {model_name} + {pipeline_name} ===")

            if pipeline is None:
                preprocessing_function = None
                rescale = 1.0 / 255.0
            else:
                def preprocessing_function(img, _pipeline=pipeline):
                    px = np.expand_dims(img, axis=0)
                    return _pipeline(px)[0]
                rescale = None

            (train_gen, val_gen), prep_train_time = time_function(
                get_data_generators,
                data_dir,
                img_size=(224, 224),
                batch_size=32,
                preprocessing_function=preprocessing_function,
                validation_split=0.2,
                rescale=rescale
            )

            model, base_model = model_builder((224, 224, 3), train_gen.num_classes)

            _, train_time = time_function(
                train_model,
                model,
                train_gen,
                val_gen,
                epochs=10,
                base_model=base_model,
                fine_tune_layers=20,
                fine_tune_epochs=10
            )

            metrics, report = evaluate_model(model, val_gen)

            results.append({
                "model": model_name,
                "experiment": pipeline_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "prep_train_time_sec": prep_train_time,
                "train_time_sec": train_time,
                "inference_time_sec": metrics["inference_time_sec"]
            })

            print(report)

    os.makedirs("logs", exist_ok=True)
    ts = timestamp()

    save_json(results, f"logs/results_{ts}.json")
    # save_csv(results, f"logs/results_{ts}.csv")

    print("\n" + "="*60)
    print("=== FINAL SUMMARY ===")
    print("="*60)
    for r in results:
        print(f"Model: {r['model']:12} | Pipeline: {r['experiment']:20} | Accuracy: {r['accuracy']:.4f} | F1: {r['f1_score']:.4f}")


if __name__ == "__main__":
    run()