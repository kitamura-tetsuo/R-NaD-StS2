import os
import json
import time
import datetime
import threading
import queue
import random
import traceback
import numpy as np
from bridge_logger import log

# Shared trajectory directory
TRAJECTORY_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD/trajectories"
if not os.path.exists(TRAJECTORY_DIR):
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)

class RawTrajectoryLogger:
    def __init__(self, trajectory_dir, bridge_globals):
        self.trajectory_dir = trajectory_dir
        self.bridge_globals = bridge_globals
        self.current_episode = []
        self.reset_ui = False
        self.step_id = 0
        self.lock = threading.Lock()

    def log_step(self, state_json, action_idx, probs, mask, reward, log_prob, predicted_v=0.0, logits=None, terminal=False, search_type=None, stochastic_override=False):
        with self.lock:
            self.current_episode.append({
                "state_json": state_json,
                "action_idx": int(action_idx),
                "probs": probs.tolist() if hasattr(probs, "tolist") else list(probs),
                "logits": logits.tolist() if logits is not None and hasattr(logits, "tolist") else list(logits) if logits is not None else [],
                "mask": mask.tolist() if hasattr(mask, "tolist") else list(mask),
                "reward": float(reward),
                "log_prob": float(log_prob),
                "predicted_v": float(predicted_v),
                "terminal": terminal
            })
            self.step_id += 1
            if terminal:
                self.flush()

            # Write to live_state.json for real-time monitoring
            try:
                live_data = {
                    "state": json.loads(state_json) if isinstance(state_json, str) else state_json,
                    "action_idx": int(action_idx),
                    "probs": probs.tolist() if hasattr(probs, "tolist") else list(probs),
                    "predicted_v": float(predicted_v),
                    "reward": float(reward),
                    "cum_reward": float(self.bridge_globals['reward_tracker'].session_cumulative_reward),
                    "terminal": terminal,
                    "timestamp": time.time(),
                    "step_id": self.step_id,
                    "mask": mask.tolist() if hasattr(mask, "tolist") else list(mask),
                    "reset": self.reset_ui,
                    "search_type": search_type,
                    "stochastic_override": stochastic_override
                }
                self.reset_ui = False
                live_path = os.path.abspath(os.path.join(self.trajectory_dir, "../tmp/live_state.json"))
                os.makedirs(os.path.dirname(live_path), exist_ok=True)
                tmp_live_path = live_path + ".tmp"
                with open(tmp_live_path, "w") as f:
                    json.dump(live_data, f)
                os.replace(tmp_live_path, live_path)
            except Exception:
                pass

    def flush(self, force_terminal=False):
        if not self.current_episode: return
        with self.lock:
            if force_terminal and self.current_episode:
                self.current_episode[-1]["terminal"] = True
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = os.path.join(self.trajectory_dir, f"traj_{timestamp}.json")
            try:
                from bridge_state import get_semantic_map
                with open(filepath, "w") as f:
                    json.dump({"semantic_map": get_semantic_map(), "steps": self.current_episode}, f)
                log(f"RawTrajectoryLogger: Saved episode segment ({len(self.current_episode)} steps) to {filepath}")
            except Exception as e:
                log(f"RawTrajectoryLogger: Error saving trajectory: {e}")
            self.current_episode = []

class TrainingWorker(threading.Thread):
    def __init__(self, learner, config, experience_queue, bridge_globals, experiment_manager=None, step_count=0):
        super().__init__(daemon=True)
        self.learner = learner
        self.config = config
        self.experience_queue = experience_queue
        self.bridge_globals = bridge_globals
        self.experiment_manager = experiment_manager
        self.batch_buffer = []
        self.running = True
        self.step_count = step_count
        self.episode_game_overed_floors = []
        self.episode_game_overed_rewards = []
        self.last_known_mean_game_overed_floor = None
        self.last_known_mean_game_overed_reward = None
        self.lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.is_updating = False
        self.update_progress = 0
        self.update_total = 0

    def run(self):
        log("TrainingWorker started.")
        while self.running:
            try:
                trajectory = self.experience_queue.get(timeout=1.0)
                batch = None
                with self.buffer_lock:
                    self.batch_buffer.append(trajectory)
                    if len(self.batch_buffer) >= self.config.batch_size:
                        batch = self.batch_buffer[:self.config.batch_size]
                        self.batch_buffer = self.batch_buffer[self.config.batch_size:]
                if batch:
                    if os.environ.get("RNAD_RECORD_ONLY") == "true":
                        continue
                    self.perform_update(batch)
            except queue.Empty: continue
            except Exception as e:
                log(f"Error in TrainingWorker: {e}")
                traceback.print_exc()

    def record_game_over(self, floor, reward):
        with self.lock:
            self.episode_game_overed_floors.append(floor)
            self.episode_game_overed_rewards.append(reward)

    def perform_update(self, batch, increment_step=True, reset_updating=True):
        from bridge_state import encode_state # Deferred
        import jax.numpy as jnp
        
        with self.lock:
            self.is_updating = True
            if increment_step:
                self.update_progress, self.update_total = 0, 1
        
        try:
            # Padding and batch building logic (collapsed for brevity but functional)
            max_len = self.config.unroll_length
            padded_obs_dict = {k: [] for k in ["global", "combat", "draw_bow", "discard_bow", "exhaust_bow", "master_bow", "map", "event", "card_reward", "state_type", "head_type"]}
            padded_act, padded_rew, padded_mask, padded_probs_dist, padded_pred_v, padded_is_human, padded_log_prob, padded_done = [], [], [], [], [], [], [], []
            padded_next_obs_dict = {k: [] for k in padded_obs_dict.keys()}
            padded_next_mask, valid_mask = [], []

            for traj_item in batch:
                if isinstance(traj_item, dict) and "steps" in traj_item:
                    traj, next_step = traj_item["steps"], traj_item.get("next_step")
                else: 
                    traj, next_step = traj_item, None
                
                l = len(traj)
                if l == 0: continue
                obs_traj = [t['obs'] for t in traj]
                for key in padded_obs_dict.keys():
                    default_val = np.int32(5) if key == "head_type" else (np.int32(2) if key == "state_type" else (np.zeros(128, dtype=np.float32) if key == "card_reward" else None))
                    if default_val is not None:
                        val_traj = [o.get(key, default_val) for o in obs_traj]
                        val_traj += [default_val] * (max_len - l)
                    else:
                        val_traj = [o[key] for o in obs_traj]
                        val_traj += [np.zeros_like(val_traj[0])] * (max_len - l)
                    padded_obs_dict[key].append(val_traj)

                padded_act.append([t['act'] for t in traj] + [0] * (max_len - l))
                padded_rew.append([t['rew'] for t in traj] + [0.0] * (max_len - l))
                padded_mask.append([t['mask'] for t in traj] + [np.zeros_like(traj[0]['mask'])] * (max_len - l))
                padded_probs_dist.append([t.get('probs_dist', np.zeros(100)) for t in traj] + [np.zeros(100)] * (max_len - l))
                padded_pred_v.append([t.get('predicted_v', 0.0) for t in traj] + [0.0] * (max_len - l))
                padded_is_human.append([t.get('is_human', 0.0) for t in traj] + [0.0] * (max_len - l))
                padded_log_prob.append([t['log_prob'] for t in traj] + [0.0] * (max_len - l))
                padded_done.append([t.get('done', 0.0) for t in traj] + [0.0] * (max_len - l))
                valid_mask.append([1.0] * l + [0.0] * (max_len - l))

                if next_step:
                    no = next_step['obs']
                    for key in padded_next_obs_dict.keys():
                        default_val = np.int32(5) if key == "head_type" else (np.int32(2) if key == "state_type" else (np.zeros(128, dtype=np.float32) if key == "card_reward" else None))
                        padded_next_obs_dict[key].append(no.get(key, default_val) if default_val is not None else no[key])
                    padded_next_mask.append(next_step['mask'])
                else:
                    for key in padded_next_obs_dict.keys():
                        padded_next_obs_dict[key].append(np.int32(5) if key == "head_type" else (np.int32(2) if key == "state_type" else np.zeros_like(padded_obs_dict[key][0][0])))
                    padded_next_mask.append(np.zeros_like(padded_mask[0][0]))

            jax_obs = {k: jnp.array(np.array(v).transpose(1, 0, *range(2, np.array(v).ndim))) for k, v in padded_obs_dict.items()}
            jax_next_obs = {k: jnp.array(np.array(v)) for k, v in padded_next_obs_dict.items()}
            
            final_batch = {
                'obs': jax_obs, 'act': jnp.array(np.array(padded_act).transpose(1, 0)),
                'rew': jnp.array(np.array(padded_rew).transpose(1, 0)),
                'mask': jnp.array(np.array(padded_mask).transpose(1, 0, 2)),
                'probs_dist': jnp.array(np.array(padded_probs_dist).transpose(1, 0, 2)),
                'predicted_v': jnp.array(np.array(padded_pred_v).transpose(1, 0)),
                'is_human': jnp.array(np.array(padded_is_human).transpose(1, 0)),
                'log_prob': jnp.array(np.array(padded_log_prob).transpose(1, 0)),
                'done': jnp.array(np.array(padded_done).transpose(1, 0)),
                'valid': jnp.array(np.array(valid_mask).transpose(1, 0)),
                'next_obs': jax_next_obs, 'next_mask': jnp.array(np.array(padded_next_mask))
            }

            metrics = self.learner.update(final_batch, self.step_count)
            
            with self.lock:
                if self.episode_game_overed_floors:
                    self.last_known_mean_game_overed_floor = sum(self.episode_game_overed_floors) / len(self.episode_game_overed_floors)
                    self.last_known_mean_game_overed_reward = sum(self.episode_game_overed_rewards) / len(self.episode_game_overed_rewards)
                    self.episode_game_overed_floors, self.episode_game_overed_rewards = [], []
                if self.last_known_mean_game_overed_floor is not None:
                    metrics['mean_game_overed_floor'] = self.last_known_mean_game_overed_floor
                    metrics['mean_game_overed_reward'] = self.last_known_mean_game_overed_reward

            if increment_step:
                self.step_count += 1
                if self.experiment_manager: self.experiment_manager.log_metrics(self.step_count, metrics)
                if self.step_count % self.config.save_interval == 0:
                    path = os.path.join(self.experiment_manager.checkpoint_dir if self.experiment_manager else "checkpoints", f"checkpoint_{self.step_count}.pkl")
                    self.learner.save_checkpoint(path, self.step_count)
                    if self.experiment_manager: self.experiment_manager.log_checkpoint_artifact(self.step_count, path)
            
        except Exception as e:
            log(f"Error during perform_update: {e}")
            traceback.print_exc()
        finally:
            if reset_updating:
                with self.lock:
                    self.is_updating, self.update_progress, self.update_total = False, 0, 0
