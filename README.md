# R-NaD-StS2

## Overview
**R-NaD-StS2** is a Godot mod for **Slay the Spire 2** that integrates the Regularized Nash Dynamics (R-NaD) Reinforcement Learning algorithm. It establishes a robust, cross-language bridge to train an AI agent using python-based reinforcement learning from within the Unity/Godot ecosystem of StS2.

## Architecture
This project is composed of three interconnected modules running inside the game process and one external control panel:

1. **`communication-mod` (C#)**: The entry point of the Slay the Spire 2 Godot mod. It intercepts game state using Harmony patches, serializes it, and sends it to the AI bridge. It handles applying the AI's chosen actions back to the game.
2. **`GDExtension` (Rust + PyO3)**: Acts as the Godot node (`AiBridge`) bridging the C# Mod and the Python ecosystem. It spins up an embedded Python interpreter configured to process game state asynchronously.
3. **`R-NaD` (Python)**: The core logic (`rnad_bridge.py`). It receives the `state_json` from the game, runs inference/training, and returns the next action. It also spawns a lightweight local HTTP server on port 8081 to accept external control commands.
4. **`Streamlit UI` (Python)**: A web-based control panel (`ui.py`) running in a separate python process. Used to toggle the learning process on and off dynamically while the game is running.

## Prerequisites
- **Godot 4.x** (Depending on Slay the Spire 2's specific version)
- **Rust Toolchain** (for compiling the GDExtension)
- **.NET SDK** (for compiling the C# mod)
- **Python 3.10+** (System python)

## How to Build & Run

### 1. Build the Rust GDExtension
```bash
cd GDExtension
cargo build
```

### 2. Build the C# Mod
```bash
cd communication-mod
dotnet build
```

### 3. Setup Python Virtual Environment and Run the UI
You need to install dependencies in a virtual environment for the Streamlit graphical UI:
```bash
cd R-NaD
python3 -m venv venv
source venv/bin/activate
pip install streamlit requests jax jaxlib dm-haiku optax mlflow numpy
streamlit run ui.py
```
*The UI will run on `http://localhost:8501`.*

### 4. Running R-NaD Training (Optional)
You can run or verify the R-NaD training process independently:
```bash
# Launch the JAX-based training loop
export PYTHONPATH=$PYTHONPATH:$(pwd)/R-NaD
python3 R-NaD/train_sts2.py --max_steps 1000
```
*Trained checkpoints will be saved in the `checkpoints/` directory.*

### 5. Launch Slay the Spire 2
Start the game with the Mod Loader enabled. The `communication-mod` will attach the `AiBridge` Godot node, start the Python daemon, and begin streaming state/receiving actions from R-NaD. Using the Streamlit UI, you can toggle the "Learning ACTIVE/INACTIVE" state.

## License
MIT License (Refer to the LICENSE file for details).
