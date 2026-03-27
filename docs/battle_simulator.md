# Battle Simulator & Validation CLI

## Overview
The `battle_simulator` is a Rust implementation of Slay the Spire 2 combat logic. It is used as a fast, deterministic oracle for MCTS and validation during training.

## Discrepancy Logging
The Python bridge (`rnad_bridge.py`) compares the internal simulator state with the actual game state after every action. If a discrepancy is found, it saves logs to `battle_simulator/discrepancy_logs/`:

- `state_before_*.json`: Game state (C# format) before the action.
- `action_*.json`: Details of the action taken (card index, target, etc.).
- `state_after_*.json`: Actual game state (C# format) after the action.

## Debugging with `vdiff`
A CLI tool `vdiff` is provided to reproduce and debug these discrepancies locally.

### How to use
Pass the three related JSON files to the tool:
```bash
cd battle_simulator
cargo run --bin vdiff -- \
  discrepancy_logs/state_before_XX.json \
  discrepancy_logs/action_XX.json \
  discrepancy_logs/state_after_XX.json
```

### Output
- `true`: The simulator's predicted outcome matches the actual game state.
- `DISCREPANCY FOUND`: A diff will be shown highlighting mismatches in HP, Energy, Block, or other fields.

## Key Files
- `battle_simulator/src/state.rs`: Main simulation logic and state transitions.
- `battle_simulator/src/creature.rs`: Player and enemy attributes/powers.
- `battle_simulator/src/card.rs`: Card specific logic.
- `battle_simulator/src/bin/vdiff.rs`: CLI source code.
