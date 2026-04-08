import os
import sys

# Add src to path if needed (assuming test is run from project root)
sys.path.append(os.path.join(os.getcwd(), 'src'))

import jax
import jax.numpy as jnp
import haiku as hk

# The script will be run from R-NaD directory which contains src/rnad.py
# So we import from rnad directly if we are in R-NaD directory
from rnad import TransformerNet, RNaDConfig

def test_transformernet_scan():
    config = RNaDConfig(
        hidden_size=64,
        num_blocks=1,
        num_heads=2,
        seq_len=4,
        unroll_length=4,
        card_vocab_size=100,
        monster_vocab_size=40,
        relic_vocab_size=300,
        power_vocab_size=280
    )
    
    num_actions = 20
    T, B = 4, 2
    
    def forward(state_dict, mask, is_training=False):
        model = TransformerNet(num_actions, config.hidden_size, config.num_blocks, config.num_heads, seq_len=config.seq_len, config=config)
        return model(state_dict, mask, is_training=is_training)
        
    network = hk.transform(forward)
    
    # Mock state_dict
    state_dict = {
        "global": jnp.zeros((T, B, 512)),
        "combat": jnp.zeros((T, B, 384)),
        "relic_ids": jnp.zeros((T, B, 30)),
        "draw_bow": jnp.zeros((T, B, config.card_vocab_size)),
        "discard_bow": jnp.zeros((T, B, config.card_vocab_size)),
        "exhaust_bow": jnp.zeros((T, B, config.card_vocab_size)),
        "master_bow": jnp.zeros((T, B, config.card_vocab_size)),
        "map": jnp.zeros((T, B, 2048)),
        "event": jnp.zeros((T, B, 128)),
        "card_reward": jnp.zeros((T, B, 128)),
        "state_type": jnp.zeros((T, B), dtype=jnp.int32),
        "head_type": jnp.zeros((T, B), dtype=jnp.int32)
    }
    mask = jnp.ones((T, B, num_actions))
    
    key = jax.random.PRNGKey(42)
    params = network.init(key, state_dict, mask)
    
    logits, value = network.apply(params, key, state_dict, mask)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
    
    assert logits.shape == (T, B, num_actions)
    assert value.shape == (T, B)
    print("Verification successful!")

if __name__ == "__main__":
    try:
        test_transformernet_scan()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
