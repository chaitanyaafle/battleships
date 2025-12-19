# Action Masking Solution for Battleship RL

## The Problem

You trained a PPO model for 200K timesteps, but when testing it with `demo_ppo.py`, the agent keeps attacking the same cells repeatedly, getting -50 rewards for invalid moves.

### Why This Happens

1. **The observation includes attack history**: The `attack_board` in the observation shows which cells have been attacked (0=unknown, 1=miss, 2=hit)

2. **But the model doesn't learn to use it**: Standard PPO has no built-in mechanism to prevent invalid actions. The agent must learn through trial and error that cells with non-zero values in `attack_board` are invalid.

3. **This is a HARD learning problem**: Even with 200K timesteps, the model may not learn the connection between observation and valid actions, especially when:
   - The state space is large (10x10 = 100 cells)
   - Invalid actions give high penalties but don't teach the model which actions ARE valid
   - The model can get stuck in local minima

## The Solution: Action Masking

**Action masking** prevents the agent from selecting invalid actions during training. This:
- Makes learning much faster (no wasted time on invalid actions)
- Guarantees the agent only learns from valid moves
- Dramatically improves sample efficiency

### Implementation

I've created two new files:

1. **`game/wrappers.py`**: Gymnasium wrappers including `ActionMaskWrapper`
2. **`training/train_ppo_masked.py`**: Training script using `MaskablePPO` from sb3-contrib

## How to Use

### 1. Install sb3-contrib

```bash
pip install sb3-contrib
```

Or update all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train with Action Masking

```bash
python training/train_ppo_masked.py --timesteps 50000
```

With action masking, you'll need **much fewer timesteps** (50K instead of 200K+) because:
- No invalid actions during training
- Every experience is useful
- Faster convergence

### 3. Test the Model

The demo script will work with masked models too:
```bash
python demo_ppo.py --model models/ppo_masked_*/final_model
```

## Expected Results

With action masking:
- **0 invalid moves** during play
- **Faster learning** (50K timesteps ≈ better than 200K without masking)
- **Better final performance** (closer to optimal ~49 moves)

Without action masking (old approach):
- Many invalid moves even after 200K+ timesteps
- Slow, inefficient learning
- May never converge to good performance

## Debug Your Current Model

To understand what your current (non-masked) model is doing:

```bash
python debug_model.py --model models/ppo_20251030_204618/final_model
```

This will show:
- What observations the model sees
- What actions it predicts
- How many invalid actions it makes
- If it's stuck in a loop

## Key Differences

| Feature | Standard PPO | Masked PPO |
|---------|-------------|------------|
| Invalid actions | Allowed (with penalty) | **Prevented** |
| Training speed | Slow | **Fast** |
| Sample efficiency | Low | **High** |
| Final performance | May not converge | **Converges reliably** |
| Timesteps needed | 200K-500K+ | **50K-100K** |

## Next Steps

1. **Clean up old models**: `rm -rf models/ppo_2025*`
2. **Train with masking**: `python training/train_ppo_masked.py --timesteps 50000`
3. **Watch it play**: `python demo_ppo.py --model models/ppo_masked_*/final_model`
4. **Compare performance**: `python training/evaluate.py --models models/ppo_masked_*/final_model --baselines`

## Technical Details

The masking function in `train_ppo_masked.py`:

```python
def mask_fn(env) -> np.ndarray:
    """Generate action mask for current state."""
    if env.state is None:
        return np.ones(env.action_space.n, dtype=bool)

    # Valid actions are cells that haven't been attacked
    attack_board = env.state.attack_board
    return (attack_board == 0).flatten()
```

This returns a boolean array where `True` = valid action, `False` = invalid action.

The `MaskablePPO` algorithm uses this mask during:
- **Training**: Only samples valid actions
- **Inference**: Only considers valid actions when selecting action

## References

- [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- [Action Masking in RL](https://arxiv.org/abs/2006.14171)
