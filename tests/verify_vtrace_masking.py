import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add the src directory to sys.path
sys.path.append(os.path.join(os.getcwd(), "R-NaD", "src"))

from rnad import v_trace

def test_vtrace_done_mask():
    T, B = 10, 1
    gamma = 0.9
    
    # Initialization
    v_tm1 = jnp.ones((T, B))
    v_tp1 = jnp.ones((T, B))
    r_t = jnp.zeros((T, B))
    rho_t = jnp.ones((T, B))
    
    # Case 1: Done mask isolates from future
    done_t = jnp.zeros((T, B)).at[5, 0].set(1.0)
    
    # Run 1: Base rewards
    vs1, _ = v_trace(v_tm1, v_tp1, r_t, rho_t, done_t, gamma=gamma)
    
    # Run 2: Change future rewards after the "done" point (at t=6)
    r_t2 = r_t.at[6, 0].set(100.0)
    vs2, _ = v_trace(v_tm1, v_tp1, r_t2, rho_t, done_t, gamma=gamma)
    
    print(f"vs1[5]: {vs1[5, 0]}")
    print(f"vs2[5]: {vs2[5, 0]}")
    
    assert jnp.allclose(vs1[5, 0], vs2[5, 0]), f"Leakage detected! vs[5] changed when future reward changed despite done[5]=1. vs1[5]={vs1[5, 0]}, vs2[5]={vs2[5, 0]}"
    print("Verification Success: No leakage across episode boundary (t=5).")

    # Case 2: Check bootstrapping at terminal state
    # If done[5]=1, vs[5] should only depend on r[5] and v_curr[5]
    # vs = v_curr + delta
    # delta = rho * (r + gamma * (1-done) * v_next - v_curr) = 1 * (r - v_curr)
    # vs = v_curr + r - v_curr = r
    r_t3 = r_t.at[5, 0].set(42.0)
    vs3, _ = v_trace(v_tm1, v_tp1, r_t3, rho_t, done_t, gamma=gamma)
    print(f"vs3[5] (should be 42.0): {vs3[5, 0]}")
    assert jnp.allclose(vs3[5, 0], 42.0), f"Bootstrap error! vs[5] should be r[5]=42.0 when done[5]=1. Got {vs3[5, 0]}"
    print("Verification Success: Correct terminal bootstrapping.")

if __name__ == "__main__":
    test_vtrace_done_mask()
