import json
import csv
import os
from datetime import datetime

def save_json(results, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def save_csv(results, filename="results.csv"):
    keys = results[0].keys()

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")