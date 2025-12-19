# CNN Architecture for Emergent Parity Learning

Testing if spatial inductive bias from CNNs enables agents to discover parity patterns without reward shaping.

## Hypothesis

**Current State:**
- MLP agents: Learn target mode ✅ | Learn parity ❌
- Parity penalty: Forces parity learning through reward shaping

**CNN Hypothesis:**
- Convolutional filters naturally detect spatial patterns (checkerboards, adjacency)
- CNN should discover parity as an emergent strategy (no reward shaping needed)

## Architecture Comparison

### MLP (Baseline)

```
Observation Dict → Flatten all → [64] → [64] → Action Logits
                    ↓
    Loses spatial structure (board is just a vector)
```

**Problem:**
- Cell (0,0) and cell (2,2) are just different positions in a flat vector
- Network can't "see" geometric patterns
- Must learn parity through trial-and-error value estimation

### CNN (New)

```
attack_board (HxW) → Conv2d(32) → Conv2d(64) → Pool → Combine with aux → [64] → Action Logits
                            ↓                    ↓
                  Detect local patterns    Detect larger patterns
                  (adjacency, 3x3)        (checkerboard, lines)
```

**Advantages:**
- **Spatial inductive bias**: Convolution inherently processes spatial relationships
- **Translation invariance**: Same pattern anywhere on board is recognized the same way
- **Hierarchical features**: Builds up from cells → lines → patterns

## Implementation Details

### Code Changes (Minimal)

1. **New file: [game/cnn_policy.py](game/cnn_policy.py)**
   - `BattleshipCNN` class - custom CNN feature extractor
   - Compatible with Stable-Baselines3's MultiInputPolicy
   - ~120 lines of code

2. **Updated: [training/train_ppo_masked.py](training/train_ppo_masked.py)**
   - Added dynamic import for custom feature extractors
   - ~10 lines added (lines 209-221)

3. **New configs:**
   - [cnn_experiment_3x3_ship2.yaml](configs/cnn_experiment_3x3_ship2.yaml)
   - [cnn_experiment_4x4_ship2.yaml](configs/cnn_experiment_4x4_ship2.yaml)

**Total code change:** ~130 lines

### CNN Architecture Spec

```python
Input: Dict observation
  - attack_board: (H, W) with values 0 (unknown), 1 (miss), 2 (hit)
  - remaining_ships: (5,) - count of each ship type remaining
  - move_count: (1,) - current move number

Processing:
  1. attack_board → Conv2d(1→32, 3x3, padding=1)  # Detect adjacency
  2. ReLU activation
  3. Conv2d(32→64, 3x3, padding=1)                # Detect patterns
  4. ReLU activation
  5. AdaptiveAvgPool2d(2x2)                       # Fixed output size
  6. Flatten → (256,)
  7. Concatenate with [remaining_ships, move_count] → (262,)
  8. Linear(262 → 128)                            # Feature vector
  9. ReLU

Output: 128-dim feature vector → Policy head [64] → Action logits
```

### Hyperparameter Changes (vs MLP baseline)

| Parameter | MLP Baseline | CNN Experiment | Reason |
|-----------|--------------|----------------|--------|
| `total_timesteps` | 500K | **2M** | More time for emergence |
| `gamma` | 0.95 | **0.995** | Long-term planning |
| `ent_coef` | 0.005 | **0.01** | More exploration |
| `batch_size` | 128 | **64** | More frequent updates |
| `net_arch` | [64, 64] | **[128, 64]** | Match baseline structure |
| `features_dim` | N/A (MLP) | **256** | Richer CNN features |
| `eval_freq` | 50K | **200K** | Longer between evals |

## Training Commands

```bash
# 3x3 board with CNN (2M timesteps, ~2.5 hours)
python training/train_ppo_masked.py \
  --config configs/cnn_experiment_3x3_ship2.yaml \
  --name cnn_3x3_ship2

# 4x4 board with CNN (2M timesteps, ~2.5 hours)
python training/train_ppo_masked.py \
  --config configs/cnn_experiment_4x4_ship2.yaml \
  --name cnn_4x4_ship2
```

## Expected Results

### Best Case ✅
- **Miss-Adjacent%**: Drops from ~50-60% (MLP) to ~30-35% (CNN)
- **Performance**: Comparable or better than MLP baseline
- **Conclusion**: Spatial inductive bias enables emergent parity!

### Moderate Case 📊
- **Miss-Adjacent%**: Drops to ~40-45% (improvement but not full parity)
- **Performance**: Similar to MLP
- **Conclusion**: CNN helps but not sufficient alone
- **Next step**: Add Phase 2 (enhanced observations with parity_mask)

### Worst Case ❌
- **Miss-Adjacent%**: Stays ~50-60% (no improvement)
- **Performance**: Similar to MLP
- **Conclusion**: 2M timesteps insufficient or parity too hard without hints
- **Next steps**: Try 5M+ timesteps, or add intrinsic motivation (RND)

## Comparison Matrix

After training, we'll have:

| Model | Architecture | Reward Shaping | Miss%  | Mean Moves | Notes |
|-------|-------------|----------------|---------|------------|-------|
| MLP Baseline | [64,64] | None | ~50-60% | 4.6-6.3 | Current baseline |
| MLP Parity | [64,64] | Penalty -5.0 | **~28-32%** | 4.3-6.6 | Forced parity learning |
| **CNN** | Conv→[64] | None | **?** | **?** | **Testing emergent parity** |
| CNN Long | Conv→[64] | None | ? | ? | If 2M not enough, try 5M |

## W&B Metrics to Monitor

### Primary Metric
- **`rollout/miss_adjacent_rate_mean`**: Target <35% for emergent parity

### Learning Dynamics
- Watch Miss% trend over 2M timesteps
- Does it plateau early (bad) or keep improving (good)?
- Compare to MLP baseline curve

### Other Metrics
- `rollout/adjacency_rate_mean`: Should stay high (>80%) - target mode preserved
- `rollout/ep_len_mean`: Should decrease if parity helps
- Variance: Should be stable throughout training

## Evaluation Commands

```bash
# After training completes
PYTHONPATH=. python training/evaluate.py \
  --models \
    models/archive/*_no_adjacency_1ship2cells/final_model.zip \
    models/parity_*/final_model.zip \
    models/cnn_*/final_model.zip \
  --auto-detect-board-size \
  --n-episodes 100
```

This will show:

```
Agent                     Board    Mean    Median   Miss%    Adj%
--------------------------------------------------------------------------------
MLP (baseline)           3x3      4.6     5.0      49.7     100.0
MLP (parity penalty)     3x3      4.3     4.0      28.5     100.0
CNN (emergent)           3x3      ?       ?        ?        ?
```

## Timeline

- **CNN 3x3**: 2M steps × 1.5 sec/1K = ~50 min
- **CNN 4x4**: 2M steps × 2.0 sec/1K = ~67 min
- **Total**: ~2 hours for both experiments

Can run in parallel on 2 terminals.

## If CNN Works → Next Steps

1. **Scale to larger boards** (5x5) - does parity transfer?
2. **Scale to multiple ships** - does dual-mode strategy emerge?
3. **Try curriculum** - 3x3 → 4x4 → 5x5 in one training run
4. **Publish findings**: "Spatial Inductive Bias Enables Emergent Parity in RL Agents"

## If CNN Doesn't Work → Next Steps

1. **Phase 2: Enhanced observations** (add parity_mask channel)
2. **Intrinsic motivation**: Add RND/curiosity bonus
3. **Much longer training**: 5M-10M timesteps
4. **Accept reward shaping**: Parity penalty is necessary

---

**Note on "Shaped"**: In all our discussions, "shaped" refers to **reward shaping** - specifically the parity penalty experiments where we add `-5.0` penalty for shooting adjacent to misses.
