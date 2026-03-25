import mlflow
import os
import shutil
from typing import Any, Dict
import logging
import time
import pandas as pd

class ExperimentManager:
    def __init__(self, experiment_name: str, checkpoint_dir: str = "checkpoints", run_id: str | None = None, log_checkpoints: bool = False):
        self.last_checkpoint_path: str | None = None
        self.last_checkpoint_step: int | None = None
        # Ensure logs go to the project root regardless of CWD
        mlflow.set_tracking_uri("file:///home/ubuntu/src/R-NaD-StS2/mlruns")
        self.client = mlflow.tracking.MlflowClient()
        
        mlflow.set_experiment(experiment_name=experiment_name)
        self.log_checkpoints = log_checkpoints

        # Force run_id to None if it's an empty string or just whitespace to trigger latest run search
        if run_id is not None and not run_id.strip():
            run_id = None

        # If no specific run_id is provided, try to resume the latest run from this experiment
        if not run_id and not mlflow.active_run():
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                try:
                    # Search for the most recently started run in this experiment
                    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
                    if not runs.empty:
                        run_id = runs.iloc[0].run_id
                        logging.info(f"ExperimentManager: No run_id provided. Auto-detected latest MLflow run to resume: {run_id}")
                except Exception as e:
                    logging.warning(f"ExperimentManager: Failed to search for existing runs: {e}")

        # Start a run explicitly if not already active
        if run_id:
            if mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
                logging.info(f"ExperimentManager: Ending current active run ({mlflow.active_run().info.run_id}) to switch to {run_id}")
                mlflow.end_run()
            
            if not mlflow.active_run():
                # Retry loop to handle temporary file locks or metadata sync delays
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        # Attempt to resume existing run
                        mlflow.start_run(run_id=run_id)
                        logging.info(f"ExperimentManager: Successfully resumed MLflow run: {run_id}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logging.warning(f"ExperimentManager: Failed to resume run {run_id} (attempt {attempt+1}/{max_retries}): {e}. Retrying in 2s...")
                            time.sleep(2)
                        else:
                            logging.error(f"ExperimentManager: CRITICAL: Could not resume run {run_id} after {max_retries} attempts: {e}. Starting a NEW run instead.")
                            mlflow.start_run()
        elif not mlflow.active_run():
            logging.info("ExperimentManager: No run_id provided or found. Starting a NEW MLflow run.")
            mlflow.start_run()

        active_run = mlflow.active_run()
        if active_run:
            self.run_id = active_run.info.run_id
            if run_id and self.run_id != run_id:
                logging.warning(f"ExperimentManager: Active run ID ({self.run_id}) does not match requested run ID ({run_id})!")
        else:
            raise RuntimeError("ExperimentManager: Failed to start or resume an MLflow run.")

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
                self.client.set_tag(self.run_id, key, str(params[key]))

        # Filter params and ensure values are loggable (int, float, string, bool)
        filtered_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                filtered_params[k] = v
            else:
                filtered_params[k] = str(v)

        if filtered_params:
            for k, v in filtered_params.items():
                self.client.log_param(self.run_id, k, v)

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
        for k, v in flat_metrics.items():
            self.client.log_metric(self.run_id, k, v, step=step)

    def log_checkpoint_artifact(self, step: int, ckpt_path: str):
        """Logs the .pkl checkpoint as an MLflow artifact and cleans up the previous one."""
        # Cleanup PREVIOUS local checkpoint if it exists and is different from the current one
        if self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path) and self.last_checkpoint_path != ckpt_path:
            try:
                os.remove(self.last_checkpoint_path)
                logging.info(f"Cleaned up previous local checkpoint: {self.last_checkpoint_path}")
            except Exception as e:
                logging.warning(f"Failed to cleanup previous local checkpoint {self.last_checkpoint_path}: {e}")

        # Cleanup PREVIOUS MLflow artifact if enabled
        if self.log_checkpoints and self.last_checkpoint_step is not None:
            try:
                run = mlflow.get_run(self.run_id)
                artifact_uri = run.info.artifact_uri
                if artifact_uri.startswith("file://"):
                    local_artifact_root = artifact_uri[7:]
                    old_artifact_dir = os.path.join(local_artifact_root, "checkpoints", f"step_{self.last_checkpoint_step}")
                    if os.path.exists(old_artifact_dir):
                        shutil.rmtree(old_artifact_dir)
                        logging.info(f"Cleaned up previous MLflow artifact directory: {old_artifact_dir}")
            except Exception as e:
                logging.warning(f"Failed to cleanup previous MLflow artifact for step {self.last_checkpoint_step}: {e}")

        if self.log_checkpoints and os.path.exists(ckpt_path):
            # Log the directory as an artifact in a 'checkpoints' folder in MLflow
            self.client.log_artifact(self.run_id, ckpt_path, artifact_path=f"checkpoints/step_{step}")
            logging.info(f"Saved checkpoint for step {step} to MLflow.")
        elif not self.log_checkpoints:
            pass # logging.info(f"Skipping checkpoint upload to MLflow for step {step} (disabled).")
        else:
            logging.warning(f"Checkpoint path {ckpt_path} does not exist.")
        
        # Update last checkpoint info
        self.last_checkpoint_path = ckpt_path
        self.last_checkpoint_step = step
