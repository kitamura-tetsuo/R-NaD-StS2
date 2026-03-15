from __future__ import annotations
import os
import re
import pickle
import logging
from typing import NamedTuple, Tuple, List, Dict, Any, Optional
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Deferred imports placeholders
jax = None
jnp = None
hk = None
optax = None

def _init_libs():
    global jax, jnp, hk, optax
    if jax is None:
        import jax as jax_mod
        import jax.numpy as jnp_mod
        import haiku as hk_mod
        import optax as optax_mod
        jax = jax_mod
        jnp = jnp_mod
        hk = hk_mod
        optax = optax_mod

class LeagueConfig(NamedTuple):
    decks: List[str] = ["standard_deck.txt"]
    rates: List[float] = [1.0]
    fixed_decks: List[str] = []

class RNaDConfig(NamedTuple):
    batch_size: int = 128
    learning_rate: float = 1e-2
    discount_factor: float = 0.99
    max_steps: int = 1000
    entropy_schedule_start: float = 0.1
    entropy_schedule_end: float = 0.01
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    hidden_size: int = 256
    num_blocks: int = 4
    log_interval: int = 1
    save_interval: int = 10
    unroll_length: int = 200
    model_type: str = "transformer" # "mlp" or "transformer"
    num_heads: int = 4
    seq_len: int = 8
    seed: int = 42
    accumulation_steps: int = 1

def v_trace(
    v_tm1: jnp.ndarray, # (T, B)
    v_tp1: jnp.ndarray, # (T, B)
    r_t: jnp.ndarray,   # (T, B)
    rho_t: jnp.ndarray, # (T, B)
    gamma: float = 0.99,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes V-trace targets and advantages (ported from reference)."""
    T = v_tm1.shape[0]
    rho_bar_t = jnp.minimum(rho_t, clip_rho_threshold)
    c_bar_t = jnp.minimum(rho_t, clip_pg_rho_threshold)

    def scan_body(carry, x):
        acc = carry
        rho_bar, c_bar, r, v_curr, v_next = x
        delta = rho_bar * (r + gamma * v_next - v_curr)
        acc = delta + gamma * c_bar * acc
        return acc, acc + v_curr

    xs = (rho_bar_t, c_bar_t, r_t, v_tm1, v_tp1)
    xs_rev = jax.tree_util.tree_map(lambda x: x[::-1], xs)
    init_acc = jnp.zeros_like(v_tm1[0])
    _, vs_rev = jax.lax.scan(scan_body, init_acc, xs_rev)
    vs = vs_rev[::-1]

    rho_pg = jnp.minimum(rho_t, clip_pg_rho_threshold)
    pg_advantages = rho_pg * (r_t + gamma * v_tp1 - v_tm1)
    return vs, pg_advantages

def loss_fn(params, fixed_params, batch, apply_fn, config: RNaDConfig, alpha_rnad: float):
    obs = batch['obs'] # Dictionary (PyTree) of (T, B, ...)
    act = batch['act'] # (T, B)
    rew = batch['rew'] # (T, B)
    mask = batch['mask'] # (T, B, num_actions)
    log_prob_bhv = batch['log_prob'] # (T, B)
    
    # Get any element to find T and B
    any_val = jax.tree_util.tree_leaves(obs)[0]
    T, B = any_val.shape[:2]
    valid_mask = batch.get('valid', jnp.ones((T, B))) # (T, B)
    
    # Flatten T and B for forward pass using PyTree map
    obs_flat = jax.tree_util.tree_map(lambda x: x.reshape(T * B, *x.shape[2:]), obs)
    mask_flat = mask.reshape(T * B, -1)
    
    logits, values = apply_fn(params, jax.random.PRNGKey(0), obs_flat, mask_flat)
    logits = logits.reshape(T, B, -1)
    values = values.reshape(T, B)
    
    fixed_logits, _ = apply_fn(fixed_params, jax.random.PRNGKey(0), obs_flat, mask_flat)
    fixed_logits = fixed_logits.reshape(T, B, -1)
    
    log_probs = jax.nn.log_softmax(logits)
    fixed_log_probs = jax.nn.log_softmax(fixed_logits)
    
    act_one_hot = jax.nn.one_hot(act, logits.shape[-1])
    log_pi_a = jnp.sum(log_probs * act_one_hot, axis=-1)
    log_pi_fixed_a = jnp.sum(fixed_log_probs * act_one_hot, axis=-1)
    
    penalty = alpha_rnad * (log_pi_a - log_pi_fixed_a)
    r_reg = rew - penalty 
    
    log_rho = log_pi_a - log_prob_bhv
    rho = jnp.exp(log_rho)
    
    v_next = jnp.concatenate([values[1:], jnp.zeros((1, B))], axis=0)
    vs, pg_adv = v_trace(values, v_next, r_reg, rho, gamma=config.discount_factor)
    
    # Mask losses by validity
    value_loss = 0.5 * jnp.sum((jax.lax.stop_gradient(vs) - values) ** 2 * valid_mask) / jnp.maximum(jnp.sum(valid_mask), 1.0)
    policy_loss = -jnp.sum(log_pi_a * jax.lax.stop_gradient(pg_adv) * valid_mask) / jnp.maximum(jnp.sum(valid_mask), 1.0)
    
    return policy_loss + value_loss, (policy_loss, value_loss)

def _get_hk_module():
    _init_libs()
    return hk.Module

class TransformerBlock: # Defined later to avoid import issues
    pass

class TransformerNet: # Defined later to avoid import issues
    pass

def combat_vec_to_id(val):
    """Helper to convert float card ID index back to int32 for embedding."""
    return jnp.round(val).astype(jnp.int32)

def _define_transformer_classes():
    global TransformerBlock, TransformerNet
    _init_libs()
    
    class _TransformerBlock(hk.Module):
        def __init__(self, num_heads, key_size, hidden_size, name=None):
            super().__init__(name=name)
            self.num_heads = num_heads
            self.key_size = key_size
            self.hidden_size = hidden_size

        def __call__(self, x, is_training=False):
            # x: (B, SeqLen, D)
            d = x.shape[-1]

            # Self-Attention
            attn_out = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init=hk.initializers.VarianceScaling(2.0),
                model_size=d,
            )(x, x, x)

            # Add & Norm
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + attn_out)

            # MLP
            mlp_out = hk.nets.MLP(
                [d * 4, d],
                activation=jax.nn.gelu
            )(x)

            # Add & Norm
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + mlp_out)
            return x

    class _CombatExpert(hk.Module):
        def __init__(self, hidden_size, num_blocks, num_heads, seq_len, name=None):
            super().__init__(name=name)
            self.hidden_size = hidden_size
            self.num_blocks = num_blocks
            self.num_heads = num_heads
            self.seq_len = seq_len
            self.card_embedding = hk.Embed(vocab_size=100, embed_dim=64)
        
        def __call__(self, h_global, combat_obs, bow_obs, is_training=False):
            # combat_obs: [batch, 256]
            # h_global: [batch, 32]
            # bow_obs: dict of vectors
            
            batch_size = combat_obs.shape[0]
            
            # 1. Process Global & BoW context
            bow_vecs = [bow_obs[k] for k in ["draw_bow", "discard_bow", "exhaust_bow", "master_bow"]]
            bow_combined = jnp.concatenate(bow_vecs, axis=-1) # [batch, 400]
            bow_proj = hk.Linear(128)(bow_combined)
            bow_proj = jax.nn.relu(bow_proj)

            context = jnp.concatenate([h_global, bow_proj], axis=-1) # [batch, 160]
            context_proj = hk.Linear(self.hidden_size)(context)
            
            # 2. Extract and Embed hand cards
            # Multi-feature embedding for hand (10 cards, 10 features each starting at index 10)
            hand_cards = []
            for i in range(10):
                base_idx = 10 + i * 10
                card_id_idx = combat_vec_to_id(combat_obs[base_idx])
                card_embed = self.card_embedding(card_id_idx) # [64]
                
                # Other features (indices 1 to 9 relative to base_idx)
                other_feats = combat_obs[base_idx+1 : base_idx+10] # [9]
                card_feat = jnp.concatenate([card_embed, other_feats], axis=-1)
                hand_cards.append(hk.Linear(self.hidden_size)(card_feat))
                
            # 3. Extract and Embed enemies
            enemy_nodes = []
            for i in range(5):
                base_idx = 110 + i * 12
                enemy_feat = combat_obs[base_idx : base_idx+12]
                enemy_nodes.append(hk.Linear(self.hidden_size)(enemy_feat))
                
            # 4. Sequence for Transformer
            # [Context, Hand x 10, Enemy x 5] -> 16 tokens
            tokens = jnp.stack([context_proj] + hand_cards + enemy_nodes, axis=0) # [16, hidden]
            
            pos_emb = hk.get_parameter("pos_emb_combat", [16, self.hidden_size], init=hk.initializers.TruncatedNormal())
            tokens = tokens + pos_emb
            
            for i in range(self.num_blocks):
                tokens = _TransformerBlock(
                    num_heads=self.num_heads,
                    key_size=self.hidden_size // self.num_heads,
                    hidden_size=self.hidden_size,
                    name=f"combat_block_{i}"
                )(tokens, is_training)
            
            return jnp.mean(tokens, axis=0) # Average pool tokens

    class _MapExpert(hk.Module):
        def __init__(self, hidden_size, num_blocks, num_heads, seq_len, name=None):
            super().__init__(name=name)
            self.hidden_size = hidden_size
            self.num_blocks = num_blocks
            self.num_heads = num_heads
            self.seq_len = seq_len
        
        def __call__(self, h_global, map_obs, is_training=False):
            # map_obs: [2048]
            # h_global: [64]
            # 1. Process Global context
            context_proj = hk.Linear(self.hidden_size)(h_global)
            
            # 2. Extract node features
            # 256 nodes, 8 features each
            node_feats = map_obs.reshape(256, 8)
            nodes_proj = hk.Linear(self.hidden_size)(node_feats) # [256, hidden]
            
            # 3. Sequence for Transformer
            # [Context, Nodes x 256] -> 257 tokens
            tokens = jnp.concatenate([context_proj[None, :], nodes_proj], axis=0) # [257, hidden]
            
            pos_emb = hk.get_parameter("pos_emb_map", [257, self.hidden_size], init=hk.initializers.TruncatedNormal())
            tokens = tokens + pos_emb
            
            for i in range(self.num_blocks):
                tokens = _TransformerBlock(
                    num_heads=self.num_heads,
                    key_size=self.hidden_size // self.num_heads,
                    hidden_size=self.hidden_size,
                    name=f"map_block_{i}"
                )(tokens, is_training)
            
            return jnp.mean(tokens, axis=0)

    class _SimpleExpert(hk.Module):
        def __init__(self, hidden_size, name=None):
            super().__init__(name=name)
            self.hidden_size = hidden_size
            
        def __call__(self, h_global, expert_obs):
            h = jnp.concatenate([h_global, expert_obs], axis=-1)
            return hk.nets.MLP([self.hidden_size * 2, self.hidden_size], name="mlp")(h)

    class _TransformerNet(hk.Module):
        def __init__(self, num_actions, hidden_size, num_blocks, num_heads, seq_len):
            super().__init__()
            self.num_actions = num_actions
            self.hidden_size = hidden_size 
            self.num_blocks = num_blocks
            self.num_heads = num_heads
            self.seq_len = seq_len
            
            # Define sub-modules here for stable parameter registration
            self.global_proj = hk.Linear(self.hidden_size, name="global_proj")
            self.combat_expert = _CombatExpert(hidden_size, num_blocks, num_heads, seq_len, name="combat_expert")
            self.map_expert = _MapExpert(hidden_size, num_blocks, num_heads, seq_len, name="map_expert")
            self.event_expert = _SimpleExpert(hidden_size, name="event_expert")
            self.grid_expert = _SimpleExpert(hidden_size, name="grid_expert")
            self.hand_expert = _SimpleExpert(hidden_size, name="hand_expert")
            self.policy_head = hk.Linear(num_actions, name="policy_head")
            self.value_head = hk.Linear(1, name="value_head")

        def __call__(self, state_dict, mask, is_training=False):
            # Process Global Backbone (shared)
            h_global = jax.vmap(lambda x: jax.nn.relu(self.global_proj(x)))(state_dict["global"])

            # Expert branches as closures
            def route_expert(st_idx, h_g, s_dict):
                bow_obs = {
                    "draw_bow": s_dict["draw_bow"],
                    "discard_bow": s_dict["discard_bow"],
                    "exhaust_bow": s_dict["exhaust_bow"],
                    "master_bow": s_dict["master_bow"]
                }
                return jax.lax.switch(st_idx, [
                    lambda: self.combat_expert(h_g, s_dict["combat"], bow_obs, is_training),
                    lambda: self.map_expert(h_g, s_dict["map"]),
                    lambda: self.event_expert(h_g, s_dict["event"]),
                    lambda: self.grid_expert(h_g, s_dict["event"]), # Reusing event vector
                    lambda: self.hand_expert(h_g, s_dict["event"])  # Reusing event vector
                ])

            # Use vmap to apply switch across batch
            if not hk.running_init():
                features = jax.vmap(route_expert)(state_dict["state_type"], h_global, state_dict)
            else:
                # During init, ensure ALL experts are initialized by calling them once
                bow_obs = {
                    "draw_bow": state_dict["draw_bow"],
                    "discard_bow": state_dict["discard_bow"],
                    "exhaust_bow": state_dict["exhaust_bow"],
                    "master_bow": state_dict["master_bow"]
                }
                self.combat_expert(h_global[0], state_dict["combat"][0], {k: v[0] for k, v in bow_obs.items()}, is_training)
                self.map_expert(h_global[0], state_dict["map"][0], is_training)
                self.event_expert(h_global[0], state_dict["event"][0])
                self.grid_expert(h_global[0], state_dict["event"][0])
                self.hand_expert(h_global[0], state_dict["event"][0])
                features = jax.vmap(route_expert)(state_dict["state_type"], h_global, state_dict)

            # Unified Heads
            logits = self.policy_head(features)
            logits = jnp.where(mask.astype(bool), logits, -1e9)
            value = self.value_head(features)
            
            return logits, jnp.squeeze(value, axis=-1)
            
    TransformerBlock = _TransformerBlock
    TransformerNet = _TransformerNet

def partial_load_params(target_params, source_params):
    """Recursively copies parameters from source to target, handling shape mismatches by slicing."""
    new_params = {}
    for key, target_val in target_params.items():
        if key not in source_params:
            print(f"[R-NaD] Parameter {key} not found in checkpoint. Using initialized values.")
            new_params[key] = target_val
            continue
            
        source_val = source_params[key]
        if isinstance(target_val, dict):
            new_params[key] = partial_load_params(target_val, source_val)
        else:
            # target_val and source_val are jnp.ndarrays
            if target_val.shape == source_val.shape:
                new_params[key] = source_val
            else:
                print(f"[R-NaD] Shape mismatch for {key}: target {target_val.shape}, source {source_val.shape}. Loading partial weights.")
                # Create a new array with target shape, initialized with target values
                merged = jnp.copy(target_val)
                
                # Compute overlap slices
                slices = []
                for t_dim, s_dim in zip(target_val.shape, source_val.shape):
                    slices.append(slice(0, min(t_dim, s_dim)))
                
                # Copy overlapping part
                merged = merged.at[tuple(slices)].set(source_val[tuple(slices)])
                new_params[key] = merged
                
    return new_params

class RNaDLearner:
    def __init__(self, state_dim: int, num_actions: int, config: RNaDConfig):
        _init_libs()
        _define_transformer_classes()
        self.config = config
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        def forward(state_dict, mask, is_training=False):
            if config.model_type == "transformer":
                model = TransformerNet(
                    num_actions=num_actions,
                    hidden_size=config.hidden_size,
                    num_blocks=config.num_blocks,
                    num_heads=config.num_heads,
                    seq_len=config.seq_len
                )
                return model(state_dict, mask, is_training=is_training)
            else:
                # Basic Categorized MLP for non-transformer case
                h_global = hk.Linear(config.hidden_size)(state_dict["global"])
                h_global = jax.nn.relu(h_global)
                
                def process_combat(state):
                     return hk.nets.MLP([config.hidden_size, config.hidden_size])(jnp.concatenate([h_global, state["combat"]], axis=-1))
                def process_map(state):
                     return hk.nets.MLP([config.hidden_size, config.hidden_size])(jnp.concatenate([h_global, state["map"]], axis=-1))
                def process_event(state):
                     return hk.nets.MLP([config.hidden_size, config.hidden_size])(jnp.concatenate([h_global, state["event"]], axis=-1))
                
                branches = [process_combat, process_map, process_event]
                features = jax.vmap(lambda idx, s: jax.lax.switch(idx, branches, s))(state_dict["state_type"], state_dict)
                
                logits = hk.Linear(num_actions)(features)
                logits = jnp.where(mask.astype(bool), logits, -1e9)
                value = hk.Linear(1)(features)
                return logits, jnp.squeeze(value, axis=-1)
            
        self.network = hk.transform(forward)
        
        base_optimizer = optax.adam(config.learning_rate)
        if config.accumulation_steps > 1:
            self.optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=config.accumulation_steps)
        else:
            self.optimizer = base_optimizer
            
        self.params = None
        self.fixed_params = None
        self.opt_state = None
        
        self._update_fn = jax.jit(self._update_pure)

    def init(self, key):
        # Create a dummy dictionary state matching the new structure
        dummy_state = {
            "global": jnp.zeros((1, 64)),
            "combat": jnp.zeros((1, 256)),
            "draw_bow": jnp.zeros((1, 100)),
            "discard_bow": jnp.zeros((1, 100)),
            "exhaust_bow": jnp.zeros((1, 100)),
            "master_bow": jnp.zeros((1, 100)),
            "map": jnp.zeros((1, 2048)),
            "event": jnp.zeros((1, 64)),
            "state_type": jnp.zeros((1,), dtype=jnp.int32)
        }
        dummy_mask = jnp.ones((1, self.num_actions))
        self.params = self.network.init(key, dummy_state, dummy_mask)
        self.fixed_params = self.params
        self.opt_state = self.optimizer.init(self.params)

    def _update_pure(self, params, fixed_params, opt_state, batch, alpha_rnad):
        def loss_wrapper(p):
            return loss_fn(p, fixed_params, batch, self.network.apply, self.config, alpha_rnad)
        (loss, aux), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux

    def update(self, batch, step: int):
        progress = min(1.0, step / self.config.max_steps)
        alpha = self.config.entropy_schedule_start + progress * (self.config.entropy_schedule_end - self.config.entropy_schedule_start)
        valid = batch.get('valid', jnp.ones((batch['obs'].shape[0], batch['obs'].shape[1])))
        self.params, self.opt_state, loss, aux = self._update_fn(self.params, self.fixed_params, self.opt_state, batch, alpha)
        return {"loss": loss, "policy_loss": aux[0], "value_loss": aux[1], "alpha": alpha}

    def save_checkpoint(self, path, step):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'params': self.params, 'fixed_params': self.fixed_params, 'opt_state': self.opt_state, 'step': step}, f)

    def load_checkpoint(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
            # Partial load params
            print(f"[R-NaD] Attempting to load partial weights from {path}")
            new_params = partial_load_params(self.params, data['params'])
            
            # Check if shapes matched exactly for opt_state compatibility
            def check_shapes_match(t1, t2):
                try:
                    leaves1 = jax.tree_util.tree_leaves(t1)
                    leaves2 = jax.tree_util.tree_leaves(t2)
                    if len(leaves1) != len(leaves2):
                        return False
                    for l1, l2 in zip(leaves1, leaves2):
                        if l1.shape != l2.shape:
                            return False
                    return True
                except:
                    return False
            
            params_changed = not check_shapes_match(self.params, data['params'])

            self.params = new_params
            if 'fixed_params' in data:
                self.fixed_params = partial_load_params(self.fixed_params, data['fixed_params'])
            else:
                self.fixed_params = self.params
            
            # Reset opt_state if parameters changed shape, as Adam/MultiSteps state depends on param shapes
            if params_changed:
                print("[R-NaD] Parameter shapes changed. Resetting optimizer state.")
                self.opt_state = self.optimizer.init(self.params)
            else:
                self.opt_state = data['opt_state']
                
            return data['step']
