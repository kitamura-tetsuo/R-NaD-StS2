# R-NaD Model Architecture Docs

This document describes the structured dictionary input and expert-based branching architecture for the R-NaD model in Slay the Spire 2.

## 1. System Overview

The model has transitioned from a single flat vector to a **Structured Dictionary Input + Task-Specific Experts (Hard-coded MoE)**. This allows for specialized processing of Combat, Map, and Event states with increased efficiency.

### Input Structure (PyTree)

The input to the model (and the output of `encode_state` in `rnad_bridge.py`) is a dictionary:

```python
{
    "global": [batch, 32],       # Shared features (Gold, HP, Floor, etc.)
    "combat": [batch, 128],      # Combat-specific (Hand cards, Enemies)
    "map": [batch, 64],          # Map-specific (Node layout)
    "event": [batch, 64],        # Event/Reward options
    "state_type": [batch],       # Index: 0=Combat, 1=Map, 2=Event, 3=Grid, 4=Hand
}
```

### 1.1 State-Specific Encoding Details (Experts 2, 3, 4)

For Event/Selection states, the `event` vector is populated:

- **Expert 2: Rewards/Event Screen**: Encodes rewards or event options.
- **Expert 3: Grid Selection (Permanent)**: Encodes deck cards for permanent changes (e.g., Upgrade at Rest Site).
- **Expert 4: Hand Selection (In-combat)**: Encodes hand cards for temporary combat effects (e.g., Armaments).
- **Encoding Structure**: Encodes up to 10 cards (Presence, ID hash, Upgraded, Cost) + Differentiation Flag at index 40.

## 2. Model Structure (`rnad.py`)

The `TransformerNet` consists of three main parts:

1.  **Global Backbone**: A shared linear layer that processes "global" features across all states.
2.  **Expert Modules**:
    - **`CombatExpert`**: A Transformer-based module for sequence processing.
    - **`SimpleExpert` (Map/Event)**: Lightweight MLP modules for non-combat logic. Handles Map, Rewards, Events, Shop, and Card Selection (Grid/Hand).
3.  **Unified Heads**: Shared `policy_head` and `value_head` that map expert features to action logits and state values.

### Execution Flow

The forward pass use `jax.lax.switch` within a `jax.vmap` batch loop:
- **Combat States** are routed to the Transformer expert.
- **Map/Event States** are routed to MLP experts.
- This saves significant computation by skipping heavy Transformer blocks during non-combat choices.

---

## 3. Extension Guidelines

### Adding a New State Type (e.g., "Shop")

If you want to add a specialized network for a new screen (like the Shop):

1.  **Modify `rnad_bridge.py`**:
    - Update `encode_state` to recognize the new state.
    - Add a new key to the dictionary (e.g., `"shop"`) and flatten relevant data into it.
    - Assign a new index to `state_type` (e.g., `3`).

2.  **Modify `rnad.py`**:
    - Create a new Expert class (e.g., `_ShopExpert`) or reuse `_SimpleExpert`.
    - Initialize the expert in `TransformerNet.__init__`.
    - Add the expert to the `jax.lax.switch` list in `TransformerNet.__call__`.
    - **Important**: Call the new expert once inside the `if hk.running_init():` block to ensure its parameters are registered.

3.  **Modify Action Mapping**:
    - Update the action decoding logic in `rnad_bridge.py`'s `predict_action` to handle any new actions specific to that screen.
