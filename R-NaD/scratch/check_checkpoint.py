import pickle
import jax
import jax.numpy as jnp
import sys

def check_checkpoint(path):
    print(f"Checking checkpoint: {path}")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        params = data.get('params')
        if not params:
            print("No params found in checkpoint.")
            return

        has_nan = False
        def check_leaf(x):
            nonlocal has_nan
            if jnp.isnan(x).any():
                has_nan = True
        
        jax.tree_util.tree_map(check_leaf, params)
        
        if has_nan:
            print("CRITICAL: Checkpoint contains NaNs!")
        else:
            print("Checkpoint is clean (no NaNs).")
            
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint.py <path>")
    else:
        check_checkpoint(sys.argv[1])
