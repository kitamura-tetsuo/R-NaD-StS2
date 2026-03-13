import os
import json

class ExperimentManager:
    def __init__(self, experiment_name="RNaD_StS2", checkpoint_dir="checkpoints", run_id=None, log_checkpoints=False):
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        self.log_checkpoints = log_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

    def log_metrics(self, metrics, step):
        print(f"Step {step}: {metrics}")

    def save_metadata(self, metadata):
        path = os.path.join(self.checkpoint_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f)
