import json
import csv
import os
from datetime import datetime


def save_json(results, filename="results.json"):
    """Serialise `results` to a formatted JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def save_csv(results, filename="results.csv"):
    """Write a list of dicts to a CSV file using the keys of the first entry as headers."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    keys = results[0].keys()

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def timestamp():
    """Return the current datetime as a compact string suitable for use in file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")