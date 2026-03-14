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
    save_interval: int = 1000
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
    obs = batch['obs'] # (T, B, dim)
    act = batch['act'] # (T, B)
    rew = batch['rew'] # (T, B)
    log_prob_bhv = batch['log_prob'] # (T, B)
    
    T, B, _ = obs.shape
    
    logits, values = apply_fn(params, jax.random.PRNGKey(0), obs.reshape(-1, obs.shape[-1]))
    logits = logits.reshape(T, B, -1)
    values = values.reshape(T, B)
    
    fixed_logits, _ = apply_fn(fixed_params, jax.random.PRNGKey(0), obs.reshape(-1, obs.shape[-1]))
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
    
    value_loss = 0.5 * jnp.mean((jax.lax.stop_gradient(vs) - values) ** 2)
    policy_loss = -jnp.mean(log_pi_a * jax.lax.stop_gradient(pg_adv))
    

    return policy_loss + value_loss, (policy_loss, value_loss)

def _get_hk_module():
    _init_libs()
    return hk.Module

class TransformerBlock: # Defined later to avoid import issues
    pass

class TransformerNet: # Defined later to avoid import issues
    pass

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

    class _TransformerNet(hk.Module):
        def __init__(self, num_actions, hidden_size, num_blocks, num_heads, seq_len):
            super().__init__()
            self.num_actions = num_actions
            self.hidden_size = hidden_size 
            self.num_blocks = num_blocks
            self.num_heads = num_heads
            self.seq_len = seq_len

        def __call__(self, x, is_training=False):
            # x: (B, ObsDim)
            if x.ndim > 2:
                x = jnp.reshape(x, (x.shape[0], -1))

            batch_size = x.shape[0]
            
            # Feature Projection to sequence
            projection_size = self.seq_len * self.hidden_size
            x = hk.Linear(projection_size)(x)
            x = jnp.reshape(x, (batch_size, self.seq_len, self.hidden_size))

            # Position Embeddings
            pos_emb = hk.get_parameter("pos_emb", [self.seq_len, self.hidden_size], init=hk.initializers.TruncatedNormal())
            x = x + pos_emb

            # Transformer Blocks
            for i in range(self.num_blocks):
                x = _TransformerBlock(
                    num_heads=self.num_heads,
                    key_size=self.hidden_size // self.num_heads,
                    hidden_size=self.hidden_size,
                    name=f"block_{i}"
                )(x, is_training)

            # Global Pooling (Mean)
            features = jnp.mean(x, axis=1)

            # Heads
            logits = hk.Linear(self.num_actions)(features)
            value = hk.Linear(1)(features)
            return logits, jnp.squeeze(value, axis=-1)
            
    TransformerBlock = _TransformerBlock
    TransformerNet = _TransformerNet

class RNaDLearner:
    def __init__(self, state_dim: int, num_actions: int, config: RNaDConfig):
        _init_libs()
        _define_transformer_classes()
        self.config = config
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        def forward(x):
            if config.model_type == "transformer":
                model = TransformerNet(
                    num_actions=num_actions,
                    hidden_size=config.hidden_size,
                    num_blocks=config.num_blocks,
                    num_heads=config.num_heads,
                    seq_len=config.seq_len
                )
                return model(x)
            else:
                net = hk.Sequential([
                    hk.Linear(config.hidden_size), jax.nn.relu,
                    hk.Linear(config.hidden_size), jax.nn.relu,
                    hk.Linear(config.hidden_size), jax.nn.relu,
                ])
                features = net(x)
                logits = hk.Linear(num_actions)(features)
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
        dummy_obs = jnp.zeros((1, self.state_dim))
        self.params = self.network.init(key, dummy_obs)
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
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.fixed_params = data['fixed_params']
            self.opt_state = data['opt_state']
            return data['step']
