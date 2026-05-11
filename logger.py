
import json
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, experiment_name="mnist_experiment"):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.json_file = self.log_dir / f"{experiment_name}.json"

    def log(self, message):
        print(message)

        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def save_results(self, results):
        with open(self.json_file, "w") as f:
            json.dump(results, f, indent=2)

        self.log(f"Results saved to: {self.json_file}")
