import mlflow
import os
from typing import Any, Dict
import logging

class ExperimentManager:
    def __init__(self, experiment_name: str, checkpoint_dir: str = "checkpoints", run_id: str = None, log_checkpoints: bool = False):
        mlflow.set_experiment(experiment_name=experiment_name)
        self.log_checkpoints = log_checkpoints

        # Start a run explicitly if not already active
        if run_id:
            if mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
                mlflow.end_run()
            if not mlflow.active_run():
                try:
                    mlflow.start_run(run_id=run_id)
                except Exception as e:
                    logging.warning(f"Failed to resume run {run_id}: {e}. Starting a new run.")
                    mlflow.start_run()
        elif not mlflow.active_run():
            mlflow.start_run()

        self.run_id = mlflow.active_run().info.run_id

        # We save checkpoints using run_id to avoid conflicts
        self.checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, self.run_id))
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logging.info(f"Initialized ExperimentManager with checkpoint dir: {self.checkpoint_dir}, log_checkpoints: {self.log_checkpoints}")

    def log_params(self, config: Any):
        """Logs configuration parameters to MLflow."""
        if hasattr(config, '_asdict'):
            params = dict(config._asdict())
        elif isinstance(config, dict):
            params = dict(config)
        else:
            try:
                params = dict(vars(config))
            except TypeError:
                params = {"config": str(config)}

        # Define keys to exclude from parameters and which ones should be tags
        # Adapted from StS2 context
        save_as_tags = ["batch_size", "unroll_length", "model_type"]

        # Set tags
        for key in save_as_tags:
            if key in params:
                mlflow.set_tag(key, str(params[key]))

        # Filter params and ensure values are loggable (int, float, string, bool)
        filtered_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                filtered_params[k] = v
            else:
                filtered_params[k] = str(v)

        if filtered_params:
            mlflow.log_params(filtered_params)

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """Logs metrics to MLflow."""
        flat_metrics = {}

        def flatten(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, prefix + k + '.')
                else:
                    try:
                        flat_metrics[prefix + k] = float(v)
                    except (ValueError, TypeError):
                        pass # Skip non-numeric metrics

        flatten(metrics)
        mlflow.log_metrics(flat_metrics, step=step)

    def log_checkpoint_artifact(self, step: int, ckpt_path: str):
        """Logs the .pkl checkpoint as an MLflow artifact."""
        if self.log_checkpoints and os.path.exists(ckpt_path):
            # Log the directory as an artifact in a 'checkpoints' folder in MLflow
            mlflow.log_artifact(ckpt_path, artifact_path=f"checkpoints/step_{step}")
            logging.info(f"Saved checkpoint for step {step} to MLflow.")
        elif not self.log_checkpoints:
            pass # logging.info(f"Skipping checkpoint upload to MLflow for step {step} (disabled).")
        else:
            logging.warning(f"Checkpoint path {ckpt_path} does not exist.")
