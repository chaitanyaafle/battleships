# Parity Learning Experiments

These experiments test whether agents can learn good parity (spacing) strategies to reduce Miss-Adjacent%.

## Hypothesis

**Current Problem**: Agents have 50-75% Miss-Adjacent rates (shooting next to previous misses)
- **Optimal**: <20% Miss-Adjacent (checkerboard/parity pattern)
- **Impact**: High Miss% → inefficient search → more moves needed

## Two Approaches

### A. Parity Reward Shaping (300K steps)
**Hypothesis**: Direct penalty for shooting adjacent to misses will teach parity faster

**Expected Results**:
- Miss-Adjacent% should drop from ~60% to <30%
- Mean moves should decrease
- Agents learn dual-mode behavior: exploit (target mode) + explore (parity)

### B. Longer Natural Training (1M steps)
**Hypothesis**: Agents might discover parity naturally with 2x more training time

**Expected Results**:
- Check if Miss-Adjacent% naturally decreases over time
- Determine if 500K timesteps was insufficient
- Baseline for comparison with reward shaping

---

## Training Commands

### Parity Reward Shaping (300K timesteps)

```bash
# 3x3 board with parity penalty
python training/train_ppo_masked.py \
  --config configs/parity_experiment_3x3_ship2.yaml \
  --name parity_3x3_ship2

# 4x4 board with parity penalty
python training/train_ppo_masked.py \
  --config configs/parity_experiment_4x4_ship2.yaml \
  --name parity_4x4_ship2
```

### Long Training Baseline (1M timesteps)

```bash
# 3x3 board, no reward shaping, 1M steps
python training/train_ppo_masked.py \
  --config configs/longrun_3x3_ship2_1M.yaml \
  --name longrun_3x3_ship2_1M

# 4x4 board, no reward shaping, 1M steps
python training/train_ppo_masked.py \
  --config configs/longrun_4x4_ship2_1M.yaml \
  --name longrun_4x4_ship2_1M
```

---

## Key Metrics to Watch (W&B)

### Primary Metrics
- **`rollout/miss_adjacent_rate_mean`**: Target <30% (currently ~60%)
- **`rollout/ep_len_mean`**: Lower is better
- **`rollout/adjacency_rate_mean`**: Should stay high (>80%)

### Training Dynamics
- Watch for Miss-Adjacent% trend over time
- Check if it plateaus or keeps improving
- Compare convergence speed: parity penalty vs natural learning

---

## Reward Structure Comparison

| Reward Type | Baseline (500K) | Parity (300K) | Long Run (1M) |
|-------------|-----------------|---------------|---------------|
| `miss_adjacent_penalty` | 0.0 | **-5.0** | 0.0 |
| `adjacency_bonus` | 0.0 | 0.0 | 0.0 |
| Total timesteps | 500K | 300K | 1M |

**Why -5.0 penalty?**
- Strong enough to discourage bad parity
- Not too strong to override target mode benefits (hit: +2.0, sink: +5.0)
- Similar magnitude to sink reward

---

## Expected Outcomes

### If Parity Penalty Works ✅
- Miss-Adjacent% drops to 20-30%
- Performance improves significantly
- Variance decreases (more consistent strategy)
- **Next step**: Apply to all board sizes/ship configs

### If Longer Training Works ✅
- Miss-Adjacent% decreases naturally over time
- 500K was just insufficient
- **Next step**: Retrain all models with 1M steps

### If Both Work ✅
- Parity penalty learns faster (300K vs 1M)
- Long training achieves similar results eventually
- **Tradeoff**: Speed (parity) vs simplicity (natural)

### If Neither Works ❌
- Problem is fundamental (network architecture, observation space)
- **Next steps**:
  - Try CNN architecture
  - Add parity hints to observations (checkerboard pattern)
  - Consider curriculum learning

---

## Evaluation Commands

```bash
# After training completes
PYTHONPATH=. python training/evaluate.py \
  --models models/parity_*/final_model.zip models/longrun_*/final_model.zip \
  --auto-detect-board-size \
  --n-episodes 100 \
  --baselines
```

Compare with baseline from archive:
```bash
PYTHONPATH=. python training/evaluate.py \
  --models \
    models/archive/*_no_adjacency_1ship2cells/final_model.zip \
    models/parity_*/final_model.zip \
    models/longrun_*/final_model.zip \
  --auto-detect-board-size \
  --n-episodes 100
```

---

## Timeline

**Parity experiments** (300K): ~45 min each
**Long run experiments** (1M): ~2.5 hours each

Can run parity experiments first to get quick results, then kick off long runs overnight.
