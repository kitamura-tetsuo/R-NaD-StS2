import argparse
import jax
import jax.numpy as jnp
import logging
import os
from src.rnad import RNaDLearner, RNaDConfig

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()
    
    config = RNaDConfig(max_steps=args.max_steps)
    learner = RNaDLearner(state_dim=128, num_actions=50, config=config)
    learner.init(jax.random.PRNGKey(42))
    
    logging.info("Starting JAX training loop...")
    for step in range(config.max_steps):
        # Dummy batch
        batch = {
            'obs': jnp.zeros((20, 32, 128)),
            'act': jnp.zeros((20, 32), dtype=jnp.int32),
            'rew': jnp.zeros((20, 32)),
            'log_prob': jnp.zeros((20, 32))
        }
        metrics = learner.update(batch, step)
        if step % 100 == 0:
            logging.info(f"Step {step}: Loss = {metrics['loss']:.4f}")

if __name__ == "__main__":
    main()
