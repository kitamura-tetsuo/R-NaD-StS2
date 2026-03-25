from __future__ import annotations
import os
import re
import pickle
import logging
import traceback
import time
import numpy as np
from typing import NamedTuple, Tuple, List, Dict, Any, Optional
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Deferred imports placeholders
jax = None
jnp = None
hk = None
optax = None
jmp = None

def _init_libs():
    global jax, jnp, hk, optax, jmp
    if jax is None:
        import jax as jax_mod
        import jax.numpy as jnp_mod
        import haiku as hk_mod
        import optax as optax_mod
        import jmp as jmp_mod
        jax = jax_mod
        jnp = jnp_mod
        hk = hk_mod
        optax = optax_mod
        jmp = jmp_mod

_init_libs()

class LeagueConfig(NamedTuple):
    decks: List[str] = ["standard_deck.txt"]
    rates: List[float] = [1.0]
    fixed_decks: List[str] = []

class RNaDConfig(NamedTuple):
    batch_size: int = 1
    accumulation_steps: int = 16
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    max_steps: int = 1000
    entropy_schedule_start: float = 0.1
    entropy_schedule_end: float = 0.01
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    log_interval: int = 1
    save_interval: int = 1
    model_type: str = "transformer" # "mlp" or "transformer"
    num_heads: int = 6
    num_blocks: int = 4
    seq_len: int = 8
    hidden_size: int = 256
    unroll_length: int = 32
    card_vocab_size: int = 100
    monster_vocab_size: int = 40
    seed: int = None

def v_trace(
    v_tm1: jnp.ndarray, # (T, B)
    v_tp1: jnp.ndarray, # (T, B)
    r_t: jnp.ndarray,   # (T, B)
    rho_t: jnp.ndarray, # (T, B)
    done_t: jnp.ndarray, # (T, B) - 1.0 if episode ends here, 0.0 otherwise
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
        rho_bar, c_bar, r, v_curr, v_next, done = x
        discount = 1.0 - done
        delta = rho_bar * (r + gamma * discount * v_next - v_curr)
        acc = delta + gamma * discount * c_bar * acc
        return acc, acc + v_curr

    xs = (rho_bar_t, c_bar_t, r_t, v_tm1, v_tp1, done_t)
    xs_rev = jax.tree_util.tree_map(lambda x: x[::-1], xs)
    init_acc = jnp.zeros_like(v_tm1[0])
    _, vs_rev = jax.lax.scan(scan_body, init_acc, xs_rev)
    vs = vs_rev[::-1]

    rho_pg = jnp.minimum(rho_t, clip_pg_rho_threshold)
    discount_t = 1.0 - done_t
    pg_advantages = rho_pg * (r_t + gamma * discount_t * v_tp1 - v_tm1)
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
    done_mask = batch.get('done', jnp.zeros((T, B))) # (T, B)
    bootstrap_value = batch.get('bootstrap_value', jnp.zeros(B)) # (B,)
    
    # Flatten T and B for forward pass using PyTree map
    obs_flat = jax.tree_util.tree_map(lambda x: x.reshape(T * B, *x.shape[2:]), obs)
    mask_flat = mask.reshape(T * B, -1)
    
    # logits, values = apply_fn(params, jax.random.PRNGKey(0), obs_flat, mask_flat)
    # logits = logits.reshape(T, B, -1)
    # values = values.reshape(T, B)
    
    # NEW: Pass T and B directly to the network
    logits, values = apply_fn(params, jax.random.PRNGKey(0), obs, mask)
    
    # fixed_logits, _ = apply_fn(fixed_params, jax.random.PRNGKey(0), obs_flat, mask_flat)
    # fixed_logits = fixed_logits.reshape(T, B, -1)
    fixed_logits, _ = apply_fn(fixed_params, jax.random.PRNGKey(0), obs, mask)
    
    log_probs = jax.nn.log_softmax(logits)
    fixed_log_probs = jax.nn.log_softmax(fixed_logits)
    
    act_one_hot = jax.nn.one_hot(act, logits.shape[-1])
    log_pi_a = jnp.sum(log_probs * act_one_hot, axis=-1)
    log_pi_fixed_a = jnp.sum(fixed_log_probs * act_one_hot, axis=-1)
    
    penalty = alpha_rnad * (log_pi_a - log_pi_fixed_a)
    r_reg = rew - penalty 
    
    log_rho = log_pi_a - log_prob_bhv
    rho = jnp.exp(log_rho)
    
    v_next = jnp.concatenate([values[1:], bootstrap_value[None, :]], axis=0)
    vs, pg_adv = v_trace(values, v_next, r_reg, rho, done_t=done_mask, gamma=config.discount_factor)
    
    # Mask losses by validity
    value_loss = 0.5 * jnp.sum((jax.lax.stop_gradient(vs) - values) ** 2 * valid_mask) / jnp.maximum(jnp.sum(valid_mask), 1.0)
    policy_loss = -jnp.sum(log_pi_a * jax.lax.stop_gradient(pg_adv) * valid_mask) / jnp.maximum(jnp.sum(valid_mask), 1.0)
    
    return policy_loss + value_loss, (policy_loss, value_loss)

# --- Transformer Model Definition (Functional Pattern for Haiku) ---

def combat_vec_to_id(val):
    return jnp.round(val).astype(jnp.int32)

class TransformerBlock(hk.Module):
    def __init__(self, num_heads, key_size, hidden_size, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.hidden_size = hidden_size

    def __call__(self, x, is_training=False):
        def _forward(x):
            d = x.shape[-1]
            attn_out = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init=hk.initializers.VarianceScaling(2.0),
                model_size=d,
            )(x, x, x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + attn_out)
            mlp_out = hk.nets.MLP([d * 4, d], activation=jax.nn.gelu)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + mlp_out)
            return x

        return hk.remat(_forward)(x)

class CombatExpert(hk.Module):
    def __init__(self, hidden_size, num_blocks, num_heads, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
    
    def __call__(self, h_global, combat_obs, bow_obs, is_training: bool = False, config: Optional[RNaDConfig] = None):
        card_vocab = config.card_vocab_size if config else 100
        card_embedding = hk.Embed(vocab_size=card_vocab, embed_dim=64, name="card_embedding")
        bow_vecs = [bow_obs[k] for k in ["draw_bow", "discard_bow", "exhaust_bow", "master_bow"]]
        bow_combined = jnp.concatenate(bow_vecs, axis=-1)
        bow_proj = jax.nn.relu(hk.Linear(128, name="bow_proj")(bow_combined))
        context = jnp.concatenate([h_global, bow_proj], axis=-1)
        context_proj = hk.Linear(self.hidden_size, name="context_proj")(context)
        
        hand_cards = []
        for i in range(10):
            base_idx = 10 + i * 10
            card_id_idx = combat_vec_to_id(combat_obs[base_idx])
            card_embed = card_embedding(card_id_idx)
            other_feats = combat_obs[base_idx+1 : base_idx+10]
            card_feat = jnp.concatenate([card_embed, other_feats], axis=-1)
            hand_cards.append(hk.Linear(self.hidden_size, name=f"hand_linear_{i}")(card_feat))
            
        enemy_nodes = []
        monster_vocab = config.monster_vocab_size if config else 40
        monster_embedding = hk.Embed(vocab_size=monster_vocab, embed_dim=32, name="monster_embedding")
        for i in range(5):
            base_idx = 110 + i * 16
            alive = combat_obs[base_idx : base_idx + 1]
            monster_id_idx = combat_vec_to_id(combat_obs[base_idx + 1])
            monster_embed = monster_embedding(monster_id_idx)
            is_minion = combat_obs[base_idx + 2 : base_idx + 3]
            other_feats = combat_obs[base_idx + 3 : base_idx + 14]
            combined_enemy_feat = jnp.concatenate([alive, monster_embed, is_minion, other_feats], axis=-1)
            enemy_nodes.append(hk.Linear(self.hidden_size, name=f"enemy_linear_{i}")(combined_enemy_feat))
            
        tokens = jnp.stack([context_proj] + hand_cards + enemy_nodes, axis=0)
        pos_emb = hk.get_parameter("pos_emb_combat", [16, self.hidden_size], init=hk.initializers.TruncatedNormal())
        tokens = tokens + pos_emb
        
        for i in range(self.num_blocks):
            tokens = TransformerBlock(self.num_heads, self.hidden_size // self.num_heads, self.hidden_size, name=f"block_{i}")(tokens, is_training)
        
        return jnp.mean(tokens, axis=0)

class MapExpert(hk.Module):
    def __init__(self, hidden_size, num_blocks, num_heads, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
    
    def __call__(self, h_global, map_obs, bow_obs, is_training: bool = False, config: Optional[RNaDConfig] = None):
        bow_vecs = [bow_obs[k] for k in ["draw_bow", "discard_bow", "exhaust_bow", "master_bow"]]
        bow_combined = jnp.concatenate(bow_vecs, axis=-1)
        bow_proj = jax.nn.relu(hk.Linear(128, name="bow_proj")(bow_combined))
        context = jnp.concatenate([h_global, bow_proj], axis=-1)
        context_proj = hk.Linear(self.hidden_size, name="context_proj")(context)

        node_feats = map_obs.reshape(256, 8)
        nodes_proj = hk.Linear(self.hidden_size, name="nodes_proj")(node_feats)
        tokens = jnp.concatenate([context_proj[None, :], nodes_proj], axis=0)
        pos_emb = hk.get_parameter("pos_emb_map", [257, self.hidden_size], init=hk.initializers.TruncatedNormal())
        tokens = tokens + pos_emb
        
        for i in range(self.num_blocks):
            tokens = TransformerBlock(self.num_heads, self.hidden_size // self.num_heads, self.hidden_size, name=f"block_{i}")(tokens, is_training)
        
        return jnp.mean(tokens, axis=0)

class SimpleExpert(hk.Module):
    def __init__(self, hidden_size, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        
    def __call__(self, h_global, expert_obs, bow_obs, is_training=False):
        bow_vecs = [bow_obs[k] for k in ["draw_bow", "discard_bow", "exhaust_bow", "master_bow"]]
        bow_combined = jnp.concatenate(bow_vecs, axis=-1)
        h = jnp.concatenate([h_global, expert_obs, bow_combined], axis=-1)
        return hk.nets.MLP([self.hidden_size * 2, self.hidden_size], name="mlp")(h)

class TemporalTransformer(hk.Module):
    def __init__(self, num_heads, key_size, hidden_size, num_blocks, seq_len, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.seq_len = seq_len

    def __call__(self, x, is_training=False):
        # x: (T, B, hidden_size)
        T, B, C = x.shape
        # logging.info(f"[Temporal] num_heads={self.num_heads}, key_size={self.key_size}, hidden_size={self.hidden_size}")
        
        # Causal mask for temporal attention
        mask = jnp.tril(jnp.ones((T, T)))
        mask = mask[None, None, :, :] # (1, 1, T, T) for Haiku MultiHeadAttention
        
        # Position embedding (up to 512 steps)
        pos_emb = hk.get_parameter("pos_emb_temporal", [512, C], init=hk.initializers.TruncatedNormal())
        x = x + pos_emb[:T, None, :]
        
        for i in range(self.num_blocks):
            # Haiku MultiHeadAttention expects (T, B, C) and mask (B, num_heads, T, T)
            # Actually hk.MultiHeadAttention expects (T, B, C) and mask of shape (B, num_heads, T, T)
            # Wait, standard Haiku MHA takes (query, key, value, mask)
            # Let's use it simply.
            # Manual Multi-Head Attention to avoid mysterious Haiku shape doubling
            query = hk.Linear(self.hidden_size, name=f"manual_q_{i}")(x)
            key = hk.Linear(self.hidden_size, name=f"manual_k_{i}")(x)
            value = hk.Linear(self.hidden_size, name=f"manual_v_{i}")(x)
            
            # Split heads: (T, B, hidden_size) -> (T, B, num_heads, key_size)
            q_heads = query.reshape(T, B, self.num_heads, -1)
            k_heads = key.reshape(T, B, self.num_heads, -1)
            v_heads = value.reshape(T, B, self.num_heads, -1)
            
            # (T, B, H, K) x (T, B, H, K) -> (B, H, T, T)
            # Transpose to (B, H, T, K) for easier dot product
            q_heads = q_heads.transpose(1, 2, 0, 3) # (B, H, T, K)
            k_heads = k_heads.transpose(1, 2, 0, 3) # (B, H, T, K)
            v_heads = v_heads.transpose(1, 2, 0, 3) # (B, H, T, K)
            
            # Dot-product attention
            # (B, H, T, K) @ (B, H, K, T) -> (B, H, T, T)
            logits = jnp.einsum("bhik,bhjk->bhij", q_heads, k_heads) / jnp.sqrt(self.key_size)
            
            # Apply causal mask
            # mask shape is (1, 1, T, T) or (T, T)
            # causal_mask = jnp.tril(jnp.ones((T, T)))
            # logits = jnp.where(causal_mask[None, None, :, :].astype(bool), logits, -1e9)
            logits = jnp.where(mask.astype(bool), logits, -1e9)
            
            weights = jax.nn.softmax(logits, axis=-1)
            
            # (B, H, T, T) @ (B, H, T, K) -> (B, H, T, K)
            attn_out_heads = jnp.einsum("bhij,bhjk->bhik", weights, v_heads)
            
            # Transpose back: (B, H, T, K) -> (T, B, H, K) -> (T, B, hidden_size)
            attn_out = attn_out_heads.transpose(2, 0, 1, 3).reshape(T, B, -1)
            attn_out = hk.Linear(self.hidden_size, name=f"manual_proj_{i}")(attn_out)
            
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"hist_ln_1_{i}")(x + attn_out)
            
            mlp_out = hk.nets.MLP([C * 4, C], activation=jax.nn.gelu, name=f"hist_mlp_{i}")(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"hist_ln_2_{i}")(x + mlp_out)
            
        return x

class TransformerNet(hk.Module):
    def __init__(self, num_actions, hidden_size, num_blocks, num_heads, seq_len=8, config=None, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.hidden_size = hidden_size 
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.config = config

    def __call__(self, state_dict, mask, is_training=False):
        # Use a more direct name and ensure it's not nested in '~'
        h_g_batch = jax.nn.relu(hk.Linear(self.hidden_size, name="hg_proj")(state_dict["global"]))

        combat_expert = CombatExpert(self.hidden_size, self.num_blocks, self.num_heads, name="combat_expert")
        map_expert = MapExpert(self.hidden_size, self.num_blocks, self.num_heads, name="map_expert")
        event_expert = SimpleExpert(self.hidden_size, name="event_expert")
        grid_expert = SimpleExpert(self.hidden_size, name="grid_expert")
        hand_expert = SimpleExpert(self.hidden_size, name="hand_expert")
        temporal_transformer = TemporalTransformer(
            num_heads=4,
            key_size=32,
            hidden_size=self.hidden_size,
            num_blocks=1,
            seq_len=self.seq_len,
            name="temporal_history_v4"
        )
        policy_heads = [hk.Linear(self.num_actions, name=f"policy_head_{i}") for i in range(6)]
        value_heads = [hk.Linear(1, name=f"value_head_{i}") for i in range(6)]

        def route_expert(st_idx, h_g, s_dict):
            bow_obs = {
                "draw_bow": s_dict["draw_bow"],
                "discard_bow": s_dict["discard_bow"],
                "exhaust_bow": s_dict["exhaust_bow"],
                "master_bow": s_dict["master_bow"]
            }
            return jax.lax.switch(st_idx, [
                lambda: combat_expert(h_g, s_dict["combat"], bow_obs, is_training, config=self.config),
                lambda: map_expert(h_g, s_dict["map"], bow_obs, is_training, config=self.config),
                lambda: event_expert(h_g, s_dict["event"], bow_obs, is_training),
                lambda: grid_expert(h_g, s_dict["event"], bow_obs, is_training),
                lambda: hand_expert(h_g, s_dict["event"], bow_obs, is_training)
            ])

        def route_head(head_idx, feat):
            # Using i=i to capture the value of i at definition time
            return jax.lax.switch(head_idx, [
                lambda i=i: (policy_heads[i](feat), value_heads[i](feat)) for i in range(6)
            ])

        # Get T and B dimensions
        any_val = jax.tree_util.tree_leaves(state_dict)[0]
        T, B = any_val.shape[:2]

        if hk.running_init():
            # Mandatory visit to all branches during initialization
            dummy_h_g = h_g_batch[0, 0] # (hidden_size,)
            dummy_bow_obs = {k: v[0, 0] for k, v in {
                "draw_bow": state_dict["draw_bow"],
                "discard_bow": state_dict["discard_bow"],
                "exhaust_bow": state_dict["exhaust_bow"],
                "master_bow": state_dict["master_bow"]
            }.items()}
            
            # Ensure every expert is called at least once during init
            _ = combat_expert(dummy_h_g, state_dict["combat"][0, 0], dummy_bow_obs, is_training, config=self.config)
            _ = map_expert(dummy_h_g, state_dict["map"][0, 0], dummy_bow_obs, is_training, config=self.config)
            _ = event_expert(dummy_h_g, state_dict["event"][0, 0], dummy_bow_obs, is_training)
            _ = grid_expert(dummy_h_g, state_dict["event"][0, 0], dummy_bow_obs, is_training)
            _ = hand_expert(dummy_h_g, state_dict["event"][0, 0], dummy_bow_obs, is_training)
            
            # Ensure every head is called at least once during init
            dummy_features = jnp.zeros(self.hidden_size)
            for i in range(6):
                _ = policy_heads[i](dummy_features)
                _ = value_heads[i](dummy_features)
            
            # Temporal transformer also needs to be called during init
            _ = temporal_transformer(jnp.zeros((1, 1, self.hidden_size)), is_training)

        # Flatten T and B for expert processing via vmap
        state_dict_flat = jax.tree_util.tree_map(lambda x: x.reshape(T * B, *x.shape[2:]), state_dict)
        h_g_batch_flat = h_g_batch.reshape(T * B, -1)
        
        # Expert processing (B dimension in vmap output will be T*B)
        features_flat = hk.vmap(route_expert, split_rng=False)(state_dict_flat["state_type"], h_g_batch_flat, state_dict_flat)
        
        # Reshape back to (T, B, hidden_size) for temporal attention
        features = features_flat.reshape(T, B, -1)
        if features.shape[-1] != self.hidden_size:
            # logging.error(f"[R-NaD] Shape mismatch: features.shape={features.shape}, expected hidden_size={self.hidden_size}")
            # Explicitly project to hidden_size if mismatch occurs as a fallback
            features = hk.Linear(self.hidden_size, name="features_fix_proj")(features)
            
        # Apply temporal transformer
        features = temporal_transformer(features, is_training)

        # Apply multi-head Actor/Critic
        # Reshape for head routing
        features_flat = features.reshape(T * B, -1)
        head_type_flat = state_dict["head_type"].reshape(T * B)
        
        logits_flat, value_flat = hk.vmap(route_head, split_rng=False)(head_type_flat, features_flat)
        
        logits = logits_flat.reshape(T, B, self.num_actions)
        logits = jnp.where(mask.astype(bool), logits, -1e9)
        value = value_flat.reshape(T, B)
        
        return logits, value

# --- End Model Definition ---

def _normalize_key(key):
    parts = key.split('/')
    normalized_parts = [p.lstrip('_') for p in parts]
    return '/'.join(normalized_parts)

def _find_matching_key(target_key, source_keys):
    if target_key in source_keys: return target_key
    prefixed_key = '__' + target_key
    if prefixed_key in source_keys: return prefixed_key
    for source_key in source_keys:
        if _normalize_key(source_key) == target_key: return source_key
        # Handle the weird ~ scope that Haiku sometimes adds
        if target_key.replace('/~/', '/') == source_key.replace('/~/', '/'): return source_key
        if target_key.replace('/~/', '/') == _normalize_key(source_key).replace('/~/', '/'): return source_key
    return None

def partial_load_params(target_params, source_params):
    if target_params is None:
        logging.warning("[R-NaD] target_params is None in partial_load_params")
        return source_params
        
    new_params = {}
    source_keys = list(source_params.keys())
    
    for key, target_val in target_params.items():
        source_key = _find_matching_key(key, source_keys)
        if source_key is None:
            new_params[key] = target_val
            continue
            
        if key != source_key:
            logging.info(f"[R-NaD] Mapping {source_key} -> {key}")
            
        source_val = source_params[source_key]
        if isinstance(target_val, dict) and isinstance(source_val, dict):
            new_params[key] = partial_load_params(target_val, source_val)
        elif isinstance(target_val, (jnp.ndarray, np.ndarray)) and isinstance(source_val, (jnp.ndarray, np.ndarray)):
            if target_val.shape == source_val.shape:
                new_params[key] = source_val
            else:
                logging.warning(f"[R-NaD] Shape mismatch for {key}: target {target_val.shape}, source {source_val.shape}. Slicing.")
                merged = jnp.array(target_val) # Ensure it is a JAX array
                slices = tuple(slice(0, min(t, s)) for t, s in zip(target_val.shape, source_val.shape))
                new_params[key] = merged.at[slices].set(source_val[slices])
        else:
            logging.warning(f"[R-NaD] Type mismatch for {key}: target {type(target_val)}, source {type(source_val)}. Using initialized values.")
            new_params[key] = target_val
    return new_params

class RNaDLearner:
    def __init__(self, state_dim: int, num_actions: int, config: RNaDConfig):
        _init_libs()
        self.config = config
        self.num_actions = num_actions
        
        # Mixed Precision setup for compute-heavy modules
        policy = jmp.Policy(
            compute_dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            output_dtype=jnp.float32,
        )
        hk.mixed_precision.set_policy(hk.Linear, policy)
        hk.mixed_precision.set_policy(hk.MultiHeadAttention, policy)
        hk.mixed_precision.set_policy(hk.nets.MLP, policy)
        
        def forward(state_dict, mask, is_training=False):
            model = TransformerNet(num_actions, config.hidden_size, config.num_blocks, config.num_heads, seq_len=config.seq_len, config=config)
            return model(state_dict, mask, is_training=is_training)
            
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
        # Dummy state with T=1, B=1 to trigger temporal blocks initialization
        dummy_state = {
            "global": jnp.zeros((1, 1, 512)),
            "combat": jnp.zeros((1, 1, 384)),
            "draw_bow": jnp.zeros((1, 1, self.config.card_vocab_size)),
            "discard_bow": jnp.zeros((1, 1, self.config.card_vocab_size)),
            "exhaust_bow": jnp.zeros((1, 1, self.config.card_vocab_size)),
            "master_bow": jnp.zeros((1, 1, self.config.card_vocab_size)),
            "map": jnp.zeros((1, 1, 2048)),
            "event": jnp.zeros((1, 1, 128)),
            "state_type": jnp.zeros((1, 1), dtype=jnp.int32),
            "head_type": jnp.zeros((1, 1), dtype=jnp.int32)
        }
        dummy_mask = jnp.ones((1, 1, self.num_actions))
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
        self.params, self.opt_state, loss, aux = self._update_fn(self.params, self.fixed_params, self.opt_state, batch, alpha)
        return {"loss": loss, "policy_loss": aux[0], "value_loss": aux[1], "alpha": alpha}

    def save_checkpoint(self, path, step):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'params': self.params, 'fixed_params': self.fixed_params, 'opt_state': self.opt_state, 'step': step}, f)

    def load_checkpoint(self, path):
        logging.info(f"[R-NaD] Loading checkpoint from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        if self.params is None:
            logging.warning("[R-NaD] self.params is None before loading. Initializing first.")
            self.init(jax.random.PRNGKey(0))
            
        new_params = partial_load_params(self.params, data['params'])
        
        # Aggressive manual mapping for known structural changes
        target_keys = list(new_params.keys())
        source_keys = list(data['params'].keys())
        
        # Try to recover hg_proj (formerly global_proj)
        hg_proj_key = None
        for tk in target_keys:
            if 'hg_proj' in tk: hg_proj_key = tk; break
            
        if hg_proj_key and hg_proj_key not in new_params: # Should be in new_params but might be initialized values
             pass 
             
        # Verification and fix-up
        for tk in target_keys:
            # If target has hg_proj but new_params has identical values to self.params (unmapped)
            # and source has global_proj, then map it.
            if 'hg_proj' in tk:
                found_source = None
                for sk in source_keys:
                    if 'global_proj' in sk:
                        found_source = sk
                        break
                if found_source:
                    logging.info(f"[R-NaD] Recovery mapping: {found_source} -> {tk}")
                    new_params[tk] = data['params'][found_source]
        
        self.params = new_params
        if 'fixed_params' in data:
            self.fixed_params = partial_load_params(self.params, data['fixed_params'])
        else:
            self.fixed_params = self.params

        if 'opt_state' in data and data['opt_state'] is not None:
            # Check if opt_state structure matches params to avoid "Dict key mismatch"
            try:
                # We do a dry-run of the optimizer update to verify tree structure compatibility
                dummy_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
                _ = self.optimizer.update(dummy_grads, data['opt_state'], self.params)
                self.opt_state = data['opt_state']
                logging.info("[R-NaD] Optimizer state loaded and verified from checkpoint.")
            except Exception as e:
                logging.warning(f"[R-NaD] Optimizer state mismatch detected ({type(e).__name__}: {e}). Re-initializing fresh.")
                self.opt_state = self.optimizer.init(self.params)
        else:
            logging.warning("[R-NaD] No optimizer state found in checkpoint. Initializing fresh.")
            self.opt_state = self.optimizer.init(self.params)
            
        return data['step']
        