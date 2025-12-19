# Adjacency Experiment Runs Guide

## Implementation Complete! ✅

### What's Been Done

1. **BattleshipEnv now supports**:
   - `custom_ships` parameter - define any ship configuration
   - `rewards` parameter - configure all reward values from YAML

2. **Training script updated**:
   - `train_ppo_masked.py` reads custom_ships and rewards from config
   - All configs ready to use

3. **8 config files created**:
   - 4x4 board: ship length 3 (with/without adjacency)
   - 4x4 board: ships lengths 2+3 (with/without adjacency)
   - 5x5 board: ship length 3 (with/without adjacency)
   - 5x5 board: ships lengths 2+3 (with/without adjacency)

---

## Experiment Sequence

Run experiments in this order:

### Phase 1: 4x4 Board, Single Ship (Length 3)

#### Run 1: No Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_4x4_ship3_no_adj.yaml \
  --name 4x4_ship3_no_adj
```

#### Run 2: With Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_4x4_ship3_with_adj.yaml \
  --name 4x4_ship3_with_adj
```

**Expected duration**: ~30-45 minutes per run (500K timesteps)

---

### Phase 2: 4x4 Board, Two Ships (Lengths 2+3)

#### Run 3: No Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_4x4_ships2_3_no_adj.yaml \
  --name 4x4_ships2_3_no_adj
```

#### Run 4: With Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_4x4_ships2_3_with_adj.yaml \
  --name 4x4_ships2_3_with_adj
```

**Expected duration**: ~30-45 minutes per run

---

### Phase 3: 5x5 Board, Single Ship (Length 3)

#### Run 5: No Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_5x5_ship3_no_adj.yaml \
  --name 5x5_ship3_no_adj
```

#### Run 6: With Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_5x5_ship3_with_adj.yaml \
  --name 5x5_ship3_with_adj
```

**Expected duration**: ~45-60 minutes per run

---

### Phase 4: 5x5 Board, Two Ships (Lengths 2+3)

#### Run 7: No Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_5x5_ships2_3_no_adj.yaml \
  --name 5x5_ships2_3_no_adj
```

#### Run 8: With Adjacency Rewards
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_5x5_ships2_3_with_adj.yaml \
  --name 5x5_ships2_3_with_adj
```

**Expected duration**: ~45-60 minutes per run

---

## Running in Parallel

You can run 2-3 experiments in parallel on your M3 Mac:

**Terminal 1:**
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_4x4_ship3_no_adj.yaml \
  --name 4x4_ship3_no_adj
```

**Terminal 2:**
```bash
python training/train_ppo_masked.py \
  --config configs/adjacency_experiment_4x4_ship3_with_adj.yaml \
  --name 4x4_ship3_with_adj
```

---

## Configuration Details

### Ship Configurations

| Experiment | Board | Ship(s) | Total Cells | % of Board |
|-----------|-------|---------|-------------|------------|
| 4x4 ship3 | 4x4 | 1x length 3 | 3/16 | 18.75% |
| 4x4 ships2_3 | 4x4 | 1x length 2, 1x length 3 | 5/16 | 31.25% |
| 5x5 ship3 | 5x5 | 1x length 3 | 3/25 | 12% |
| 5x5 ships2_3 | 5x5 | 1x length 2, 1x length 3 | 5/25 | 20% |

### Reward Configurations

**No Adjacency** (4 runs):
- `adjacency_bonus: 0.0`
- `missed_adjacency_penalty: 0.0`

**With Adjacency** (4 runs):
- `adjacency_bonus: 25.0`
- `missed_adjacency_penalty: -15.0`

**Common rewards** (all 8 runs):
- `miss: -1.0`
- `hit: 2.0`
- `sink: 5.0`
- `win: 5.0`
- `time_penalty: -0.3`
- `escalation_threshold: 15`
- `escalation_rate: -2.0`

### Training Parameters (All Runs)

- **Timesteps**: 500,000
- **Algorithm**: Masked PPO
- **Network**: [64, 64] hidden layers, ReLU
- **Learning rate**: 3e-4
- **Batch size**: 128
- **N_steps**: 2048
- **Gamma**: 0.95

---

## Evaluation After Training

After training completes, evaluate all models:

```bash
python training/evaluate.py \
  --models models/4x4_*/final_model.zip models/5x5_*/final_model.zip \
  --auto-detect-board-size \
  --baselines \
  --n-episodes 100 \
  --output results/adjacency_ship_complexity_results.csv
```

This will:
- Auto-detect board sizes for each model
- Group models by board size (4x4 and 5x5)
- Evaluate each on appropriate environment
- Compare to Random and Probability baselines
- Save combined results to CSV

---

## Expected Results

### Hypotheses to Test

1. **Ship length impact**: Does longer ship (length 3 vs 2) change adjacency learning?
2. **Multiple ships**: Does having 2 ships make adjacency more important?
3. **Board size scaling**: How does 5x5 compare to 4x4 and 3x3 (previous experiments)?
4. **Reward shaping necessity**: At what complexity does adjacency bonus become helpful?

### Success Metrics

- **Win rate**: Should be 100% (all runs)
- **Mean moves**: Lower is better
- **Median moves**: Primary comparison metric
- **Std dev**: Lower = more consistent strategy

---

## Troubleshooting

### If training fails to start:

1. **Check config file syntax**:
   ```bash
   python -c "import yaml; yaml.safe_load(open('configs/adjacency_experiment_4x4_ship3_no_adj.yaml'))"
   ```

2. **Test environment creation**:
   ```python
   from game.env import BattleshipEnv
   env = BattleshipEnv(
       board_size=(4, 4),
       custom_ships={'cruiser': 3},
       rewards={'adjacency_bonus': 0.0}
   )
   print("Environment created successfully!")
   ```

3. **Verify imports**:
   ```bash
   python -c "from sb3_contrib import MaskablePPO; print('SB3-contrib OK')"
   ```

### If models don't learn:

- Check W&B dashboard for learning curves
- Increase `ent_coef` to 0.01 for more exploration
- Ensure valid ship configurations (no overlap, fits on board)

---

## Next Steps After Completion

1. **Analyze results** using auto-detect evaluation
2. **Compare learning curves** in W&B
3. **Update EXPERIMENTS.md** with new findings
4. **Visualize** attack patterns for best models
5. **Consider** even larger boards (6x6, 7x7) or more ships

---

**Total experiment time**: ~6-8 hours (if run sequentially)
**Parallel execution**: ~2-3 hours (running 2-3 at a time)

Good luck with your experiments! 🚀
