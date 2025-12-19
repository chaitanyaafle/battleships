# Adjacency Experiment Training Commands

This file contains all commands to run the 12 experiments testing adjacency reward shaping across different board sizes and ship configurations.

## Experiment Overview

- **Total Experiments**: 12 (6 configurations × 2 reward variants)
- **Training Time**: ~500K timesteps each
- **Parallel Runs**: 2-3 recommended based on CPU capacity
- **W&B Project**: `battleship-rl-adjacency`

## Metrics Tracked

- Episode length (min, max, std)
- Episode reward (min, max, std)
- **Adjacency rate**: How often agent follows up on hits (target mode)
- **Miss-adjacency rate**: How often agent shoots adjacent to misses (inefficient parity)

---

## 3x3 Board - Ship Length 2 (Baseline Recreation)

```bash
# No adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_3x3_ship2_no_adj.yaml --name 3x3_no_adjacency_1ship2cells

# With adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_3x3_ship2_with_adj.yaml --name 3x3_with_adjacency_1ship2cells
```

---

## 4x4 Board - Ship Length 2 (Baseline Recreation)

```bash
# No adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_4x4_ship2_no_adj.yaml --name 4x4_no_adjacency_1ship2cells

# With adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_4x4_ship2_with_adj.yaml --name 4x4_with_adjacency_1ship2cells
```

---

## 4x4 Board - Ship Length 3

```bash
# No adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_4x4_ship3_no_adj.yaml --name 4x4_no_adjacency_1ship3cells

# With adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_4x4_ship3_with_adj.yaml --name 4x4_with_adjacency_1ship3cells
```

---

## 4x4 Board - Two Ships (Lengths 2 + 3)

```bash
# No adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_4x4_ships2_3_no_adj.yaml --name 4x4_no_adjacency_2ships

# With adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_4x4_ships2_3_with_adj.yaml --name 4x4_with_adjacency_2ships
```

---

## 5x5 Board - Ship Length 3

```bash
# No adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_5x5_ship3_no_adj.yaml --name 5x5_no_adjacency_1ship3cells

# With adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_5x5_ship3_with_adj.yaml --name 5x5_with_adjacency_1ship3cells
```

---

## 5x5 Board - Two Ships (Lengths 2 + 3)

```bash
# No adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_5x5_ships2_3_no_adj.yaml --name 5x5_no_adjacency_2ships

# With adjacency rewards
python training/train_ppo_masked.py --config configs/adjacency_experiment_5x5_ships2_3_with_adj.yaml --name 5x5_with_adjacency_2ships
```

---

## Evaluation (After Training Completes)

```bash
# Evaluate all models with auto-detected board sizes
python training/evaluate.py --models models/adjacency_experiment/*/final_model.zip --auto-detect-board-size --n-episodes 100
```

---

## Running Experiments in Parallel

### Option 1: Terminal Tabs/Windows
Open multiple terminal windows and run 2-3 experiments simultaneously.

### Option 2: Background Jobs (Unix/Mac)
```bash
# Run in background with nohup
nohup python training/train_ppo_masked.py --config configs/adjacency_experiment_3x3_ship2_no_adj.yaml --name 3x3_no_adjacency_1ship2cells > logs/3x3_no_adj.log 2>&1 &

nohup python training/train_ppo_masked.py --config configs/adjacency_experiment_3x3_ship2_with_adj.yaml --name 3x3_with_adjacency_1ship2cells > logs/3x3_with_adj.log 2>&1 &

# Monitor progress
tail -f logs/3x3_no_adj.log
```

### Option 3: Screen/Tmux Sessions
```bash
# Create tmux session
tmux new -s exp1
# Run command
python training/train_ppo_masked.py --config configs/adjacency_experiment_3x3_ship2_no_adj.yaml --name 3x3_no_adjacency_1ship2cells
# Detach: Ctrl+B, then D

# Create another session
tmux new -s exp2
# Run another experiment...
```

---

## Expected Model Outputs

Models will be saved to:
```
models/adjacency_experiment/
├── 3x3_no_adjacency_1ship2cells/
├── 3x3_with_adjacency_1ship2cells/
├── 4x4_no_adjacency_1ship2cells/
├── 4x4_with_adjacency_1ship2cells/
├── 4x4_no_adjacency_1ship3cells/
├── 4x4_with_adjacency_1ship3cells/
├── 4x4_no_adjacency_2ships/
├── 4x4_with_adjacency_2ships/
├── 5x5_no_adjacency_1ship3cells/
├── 5x5_with_adjacency_1ship3cells/
├── 5x5_no_adjacency_2ships/
└── 5x5_with_adjacency_2ships/
```

Each directory contains:
- `final_model.zip` - Trained model
- `config.yaml` - Configuration used
- Checkpoints every 50K timesteps

---

## Monitoring Progress

### W&B Dashboard
View real-time metrics at: https://wandb.ai/your-username/battleship-rl-adjacency

Key metrics to watch:
- `rollout/adjacency_rate_mean` - Target mode exploitation
- `rollout/miss_adjacent_rate_mean` - Parity/efficiency (lower is better)
- `rollout/ep_len_mean` - Average moves to win (lower is better)

### TensorBoard (Local)
```bash
tensorboard --logdir logs/
```

---

## Notes

- Each experiment runs for 500K timesteps
- Evaluation every 50K timesteps (10 episodes)
- Checkpoints saved every 50K timesteps
- Network architecture: [64, 64] with ReLU activation
- All experiments use same PPO hyperparameters for fair comparison
- Miss-adjacency tracking measures parity learning (new feature)
