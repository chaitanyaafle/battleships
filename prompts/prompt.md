# Battleship AI Research Project: Complete Implementation Plan

## 🎯 Project Mission

Build a rigorous AI research study comparing RL agents, LLM agents, and optimal heuristics across varying game complexity, culminating in:

1. **Experimental Rigor**: Publication-quality analysis with statistical testing, confidence intervals, and reproducible methodology
2. **Agent Diversity**: PPO, DQN, A2C, Rainbow DQN + multiple LLMs (Claude, Llama, Mistral) + optimal probability baseline
3. **Public Impact**: Interactive website where people can play against agents, explore analysis, with data collection

**Core Philosophy**: Do minimal things, but do them absolutely right, with maximum rigor.

---

## ⏱️ Timeline & Resources

- **Duration**: 6-8 weeks full-time (~200 hours)
- **Budget**: $100-150 total
  - API costs: $50-100 (Claude + GPT, strategic use with prompt caching)
  - Compute: <$50 (Colab Pro + Lambda Labs spot instances)
  - Hosting: $0 (GitHub Pages + Netlify free tier)
- **Compute**: Cloud GPU (Colab Pro, Lambda Labs) + CPU fallback
- **Output**: GitHub Pages website + conference-quality blog post/paper

---

## 📋 Phase 1: Infrastructure & Experiment Framework (Week 1, ~30 hrs)

**Goal**: Build foundation for rigorous experimentation BEFORE implementing agents. Infrastructure-first approach ensures every experiment is tracked, reproducible, and analyzable.

### 1.1 Experiment Tracking System (8 hrs)

**Tools**: Weights & Biases (W&B) or MLflow

**Implementation**:
```python
# experiments/tracking.py
import wandb

def init_experiment(config):
    wandb.init(
        project="battleship-ai-comparison",
        config={
            "agent_type": config.agent,
            "board_size": config.board_size,
            "ship_config": config.ships,
            "rule_variant": config.rules,
            "seed": config.seed,
        }
    )

def log_game_metrics(game_result):
    wandb.log({
        "moves_to_completion": game_result.moves,
        "hit_rate": game_result.hits / game_result.moves,
        "ships_per_move": game_result.ships_sunk / game_result.moves,
        "trajectory_heatmap": wandb.Image(game_result.heatmap),
    })
```

**Tracked Metrics**:
- **Per-game**: moves to completion, hit rate, ships sunk per move, trajectory heatmaps
- **Per-agent**: win rate, mean/median/std moves, sample efficiency (for RL)
- **Per-complexity**: performance degradation curves across board sizes, ship configs
- **Training**: reward curves, loss, gradient norms (RL), API costs (LLM)

**Outputs**: Auto-logged to W&B dashboard + local JSON/CSV for offline analysis

### 1.2 Complexity Configuration System (6 hrs)

**Extend** `game/config.py`:

```python
# experiments/complexity_configs.yaml
experiments:
  board_sizes:
    - [5, 5]
    - [6, 6]
    - [7, 7]
    - [8, 8]
    - [10, 10]
    - [12, 12]

  ship_configurations:
    minimal:  # 2 ships
      - {name: "Destroyer", length: 2}
      - {name: "Cruiser", length: 3}

    standard:  # Current implementation
      - {name: "Destroyer", length: 2}
      - {name: "Cruiser", length: 3}
      - {name: "Submarine", length: 3}
      - {name: "Battleship", length: 4}
      - {name: "Carrier", length: 5}

    maximal:  # 8 ships for large boards
      - {name: "Patrol Boat", length: 2}
      - {name: "Destroyer", length: 2}
      - {name: "Submarine", length: 3}
      - {name: "Cruiser", length: 3}
      - {name: "Battleship", length: 4}
      - {name: "Carrier", length: 5}
      - {name: "Dreadnought", length: 6}
      - {name: "Supercarrier", length: 7}

  rule_variants:
    standard:
      diagonal_touching: false
      partial_info: false

    touching_allowed:
      diagonal_touching: true
      partial_info: false

    fog_of_war:
      diagonal_touching: false
      partial_info: true  # Don't reveal ship sinking immediately

  opponent_types:
    - "random"
    - "probability"
    - "ppo_trained"
    - "dqn_trained"
```

**Implementation**: Config loader that generates all experimental conditions

### 1.3 Batch Evaluation Framework (10 hrs)

```python
# experiments/run_evaluation.py
class BatchEvaluator:
    def __init__(self, agents, configs, n_games=100, seeds=range(10)):
        self.agents = agents
        self.configs = configs
        self.n_games = n_games
        self.seeds = seeds

    def run_full_evaluation(self):
        """Run N games per (agent, complexity, seed) tuple"""
        results = []
        for agent in self.agents:
            for config in self.configs:
                for seed in self.seeds:
                    games = self.run_games(agent, config, seed, self.n_games)
                    results.append({
                        'agent': agent.name,
                        'config': config,
                        'seed': seed,
                        'games': games,
                        'stats': self.compute_statistics(games)
                    })
        return results

    def compute_statistics(self, games):
        """Bootstrap confidence intervals, effect sizes"""
        return {
            'mean': np.mean([g.moves for g in games]),
            'median': np.median([g.moves for g in games]),
            'ci_95': bootstrap_ci([g.moves for g in games]),
            'hit_rate': np.mean([g.hit_rate for g in games]),
        }
```

**Statistical Testing**:
- Bootstrap confidence intervals (95%)
- Paired t-tests for agent comparison
- Effect sizes (Cohen's d)
- Multiple comparison correction (Bonferroni)

**Reproducibility**:
- Seed management: fixed seeds for each experiment
- Environment versioning: pin Gymnasium, SB3, PyTorch versions
- Dependency freezing: `requirements-exact.txt` with versions

### 1.4 Analysis Pipeline Skeleton (6 hrs)

```
analysis/
├── metrics.py              # Metric computation functions
├── statistical_tests.py    # Hypothesis testing utilities
├── visualization.py        # Plot generation (Seaborn/Plotly)
└── report_generator.py     # Auto-generate markdown/LaTeX reports
```

**Key Functions**:
```python
# analysis/metrics.py
def compute_performance_metrics(game_results):
    """Compute all metrics from game results"""
    pass

# analysis/statistical_tests.py
def compare_agents(agent_a_results, agent_b_results):
    """Paired t-test + effect size"""
    pass

# analysis/visualization.py
def plot_performance_vs_complexity(results, agents):
    """Performance curves with CI bands"""
    pass

# analysis/report_generator.py
def generate_analysis_report(all_results):
    """Auto-generate markdown report with figures"""
    pass
```

**Deliverables**:
- ✅ W&B/MLflow integration
- ✅ Complexity config system (6 board sizes × 3 ship configs × 3 rule variants = 54 conditions)
- ✅ Batch evaluation framework
- ✅ Analysis pipeline ready to plug in agents

---

## 🤖 Phase 2: RL Agents Implementation (Weeks 2-4, ~75 hrs)

**Goal**: Implement 4 RL algorithms with rigorous training, transfer learning, and ablations.

### 2.1 Environment Optimization for RL (10 hrs)

**Observation Engineering**:
```python
# game/observation_spaces.py
class EnhancedObservation:
    """Rich observation space for RL agents"""

    def __init__(self, env):
        self.env = env

    def get_observation(self):
        return {
            # Core
            'attack_board': self.env.attack_board,  # (10, 10)
            'remaining_ships': self.env.remaining_ships,  # (5,)
            'move_count': self.env.move_count,  # (1,)

            # Enhanced features
            'probability_density': self.compute_probability_layer(),  # (10, 10)
            'hit_context': self.get_adjacent_to_hits(),  # (10, 10) binary mask
            'parity_checkerboard': self.get_parity_mask(),  # (10, 10) binary
        }

    def compute_probability_layer(self):
        """Precomputed probability density (like ProbabilityAgent)"""
        # This gives RL agents access to optimal information
        pass
```

**Reward Shaping Experiments**:
- Sparse: only on hits/sinks/win
- Dense: every move gets small signal
- Shaped: dense + bonus for hunt→target transition

**Vectorized Environments**:
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# 8 parallel environments for faster training
envs = SubprocVecEnv([lambda: BattleshipEnv() for _ in range(8)])
```

### 2.2 Agent Implementations (40 hrs, ~10 hrs each)

**A. PPO (Primary Baseline)**
```python
# training/train_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

model = PPO(
    "MultiInputPolicy",  # For Dict observation space
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/ppo/"
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/ppo/",
    name_prefix="ppo_battleship"
)

model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback
)
```

**Training Protocol** (per agent):
1. Train on 10×10 standard board (500K-1M steps, ~2-4 hours on GPU)
2. Hyperparameter sweep using Optuna:
   - Learning rate: [1e-5, 1e-3]
   - Network size: [64, 128, 256] hidden units
   - Batch size: [32, 64, 128]
3. Save checkpoints every 50K steps
4. Early stopping based on validation win rate

**B. DQN (Value-Based)**
```python
# training/train_dqn.py
from stable_baselines3 import DQN

model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=10000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./logs/dqn/"
)
```

**C. A2C (Fast Alternative)**
```python
# training/train_a2c.py
from stable_baselines3 import A2C

model = A2C(
    "MultiInputPolicy",
    env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/a2c/"
)
```

**D. Rainbow DQN (Advanced)**
```python
# training/train_rainbow.py
# Implement using SB3 Contrib or custom implementation
# Features: Dueling, Double DQN, Prioritized Replay, Noisy Nets
```

### 2.3 Transfer Learning Experiments (15 hrs)

**Setup**:
1. Train on 8×8 board to convergence (~500K steps)
2. Evaluate zero-shot on 5×5, 6×6, 7×7, 10×10, 12×12
3. Fine-tune on each target size (~100K steps)
4. Measure performance degradation vs sample efficiency

**Metrics**:
- Zero-shot transfer: performance on new size without fine-tuning
- Fine-tuning efficiency: steps needed to reach 90% of specialized model
- Generalization gap: specialized model vs transferred model

**Curriculum Learning**:
```python
# training/curriculum_learning.py
class CurriculumSchedule:
    def __init__(self):
        self.stages = [
            {"board_size": (5, 5), "steps": 200000},
            {"board_size": (7, 7), "steps": 200000},
            {"board_size": (10, 10), "steps": 300000},
            {"board_size": (12, 12), "steps": 300000},
        ]

    def train_with_curriculum(self, agent):
        for stage in self.stages:
            env = BattleshipEnv(board_size=stage["board_size"])
            agent.set_env(env)
            agent.learn(total_timesteps=stage["steps"])
```

### 2.4 Ablation Studies (10 hrs)

**Dimensions to test**:
1. **Network Architecture**:
   - MLP (baseline)
   - CNN (spatial inductive bias)
   - Attention (for hit context)

2. **Observation Space**:
   - Minimal: just attack_board
   - Standard: + remaining_ships, move_count
   - Rich: + probability_density, hit_context

3. **Reward Structure**:
   - Sparse: only win/lose
   - Dense: hit/miss/sink
   - Shaped: + hunt→target bonus

**Analysis**: Which components contribute most to performance?

**Deliverables**:
- ✅ 4 trained RL agents with saved models
- ✅ Training curves logged to W&B
- ✅ Transfer learning results across 6 board sizes
- ✅ Ablation study showing critical components

---

## 🧠 Phase 3: LLM Agents Implementation (Weeks 4-5, ~40 hrs)

**Goal**: Implement LLM agents with strategic prompting, optimize for $50-100 budget.

### 3.1 LLM Agent Architecture (10 hrs)

```python
# game/agents/llm_agent.py
from anthropic import Anthropic
import json

class LLMAgent(BattleshipAgent):
    def __init__(self, model_name="claude-sonnet-4.5", use_cache=True):
        super().__init__(f"LLM-{model_name}")
        self.client = Anthropic()
        self.model = model_name
        self.use_cache = use_cache
        self.conversation_history = []

    def select_action(self, observation):
        # Serialize state
        state_text = self._serialize_board(observation)

        # Create prompt
        prompt = self._create_prompt(state_text, observation)

        # Call LLM with caching
        response = self._call_llm(prompt)

        # Parse action
        action = self._parse_response(response)

        return action

    def _serialize_board(self, obs):
        """Convert observation to text grid"""
        board = obs['attack_board']
        symbols = {0: '·', 1: '○', 2: '✕'}

        lines = ["   " + " ".join([chr(65+i) for i in range(board.shape[1])])]
        for i, row in enumerate(board):
            line = f"{i+1:2} " + " ".join([symbols[cell] for cell in row])
            lines.append(line)

        return "\n".join(lines)

    def _create_prompt(self, state_text, obs):
        """Create structured prompt with caching"""
        system_prompt = """You are playing Battleship. Your goal is to sink all enemy ships in minimum moves.

GAME RULES (CACHE THIS):
- 10×10 grid with coordinates A1-J10
- Fleet: Carrier(5), Battleship(4), Cruiser(3), Submarine(3), Destroyer(2)
- Ships cannot touch (including diagonally)
- · = unknown, ○ = miss, ✕ = hit

STRATEGY:
1. Use probability: cells with more possible ship placements are better targets
2. Target mode: When you hit, check adjacent cells (not diagonals)
3. Parity: Ships of length 2+ mean you can skip checkerboard pattern initially"""

        user_prompt = f"""Current board state:
{state_text}

Remaining ships: {obs['remaining_ships']}
Move count: {obs['move_count'][0]}

Think step-by-step:
1. What ships remain unsunk?
2. Where are current hits (✕)?
3. What cells have highest probability?
4. What's the optimal target?

Respond in JSON:
{{
    "reasoning": "Your analysis",
    "target": "A5"  // Letter + Number
}}"""

        return {
            "system": system_prompt,  # Cached across calls
            "user": user_prompt,
        }

    def _call_llm(self, prompt):
        """Call with prompt caching (85% cost reduction)"""
        messages = [{
            "role": "user",
            "content": prompt["user"]
        }]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=[{
                "type": "text",
                "text": prompt["system"],
                "cache_control": {"type": "ephemeral"}  # Cache system prompt
            }],
            messages=messages
        )

        return response.content[0].text

    def _parse_response(self, response):
        """Extract action from JSON response"""
        try:
            data = json.loads(response)
            coord = data["target"]
            # Convert "A5" → action index
            col = ord(coord[0].upper()) - ord('A')
            row = int(coord[1:]) - 1
            return row * 10 + col
        except:
            # Fallback: random valid move
            return self._random_valid_action()
```

### 3.2 Prompt Engineering (8 hrs)

**Strategies to test**:

1. **Zero-shot**: Simple "Here's the board, select a cell"
2. **Few-shot**: Include 2-3 example games with expert reasoning
3. **Chain-of-thought**: Explicit "think step-by-step" instruction
4. **Structured output**: Force JSON response for reliable parsing

**Example Few-Shot**:
```python
FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Board: All unknown (·)
Reasoning: No hits yet. Use probability. Center cells have more possible ship placements. Start with E5.
Target: E5
Result: Miss

EXAMPLE 2:
Board: E5=miss, F5=hit
Reasoning: Hit at F5! Check adjacent cells (not diagonal). Possible targets: E5(miss), F4, F6, G5. Try F6 (vertical) or G5 (horizontal).
Target: G5
Result: Hit! Ship not sunk yet.

EXAMPLE 3:
Board: F5=hit, G5=hit, H5=hit (ship sunk - Cruiser)
Reasoning: Cruiser sunk (3 cells). Remaining: Carrier(5), Battleship(4), Submarine(3), Destroyer(2). Return to hunt mode. Try high-probability cells away from the sunk ship.
Target: C3
"""
```

### 3.3 LLM Implementations (20 hrs)

**Phase 3.3a: Open Source Models (Free, 12 hrs)**

```python
# game/agents/llm_opensource.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class OpenSourceLLMAgent(BattleshipAgent):
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__(f"LLM-{model_name.split('/')[-1]}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def select_action(self, observation):
        prompt = self._create_prompt(observation)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_response(response)
```

**Models to test** (run on Colab GPU, free):
- Llama 3.1 8B/70B (Meta)
- Mistral 7B (Mistral AI)
- Qwen 2.5 7B (Alibaba)

**Budget**: 100 games per model × 3 models = 300 games (free, just GPU time)

**Phase 3.3b: Claude API (Strategic, 8 hrs)**

**Budget Optimization**:
- $50-100 = ~1.67M-3.33M input tokens at $3/MTok
- With prompt caching (85% discount): ~11M-22M cached tokens
- Average game: ~200 tokens/move × 50 moves = 10K tokens/game
- With caching: ~1.5K fresh tokens/game (system prompt cached)
- **Estimated games**: $100 / ($0.003/1K × 1.5K) ≈ 22,000 games possible!

**Actual Strategy**:
- Test prompts: 20 games × 5 prompt variants = 100 games (~$0.50)
- Main evaluation: 30 games × 6 board sizes = 180 games (~$1.80)
- Buffer: remaining $97-98 for extended experiments

**Models**:
- Claude Sonnet 4.5 (primary)
- GPT-4o (comparison if budget allows)

### 3.4 LLM Analysis (2 hrs)

**Quantitative**:
- Win rate vs RL agents
- Moves to completion vs baselines
- Hit rate, targeting efficiency
- Cost per game vs performance

**Qualitative**:
- Reasoning quality: categorize reasoning patterns
  - Probabilistic (uses ship placement logic)
  - Deterministic (simple rules)
  - Random (no clear strategy)
- Error patterns: invalid moves, redundant attacks, poor targeting
- Creativity: novel strategies not in baselines

**Deliverables**:
- ✅ 3 open-source LLM agents (Llama, Mistral, Qwen)
- ✅ 1 Claude API agent with prompt caching
- ✅ 5 prompt variants tested
- ✅ Cost-effectiveness analysis
- ✅ Qualitative reasoning analysis

---

## 📊 Phase 4: Comprehensive Analysis (Week 6, ~30 hrs)

**Goal**: Publication-quality analysis answering key research questions.

### 4.1 Core Research Questions (2 hrs - define metrics)

1. **Scaling**: How does performance degrade with board size/complexity?
2. **Sample Efficiency**: Training data needed (RL) vs zero-shot (LLM)?
3. **Strategy Comparison**: Do RL agents learn probability-based strategies?
4. **Failure Modes**: Where/why does each agent type fail?
5. **Generalization**: Which agents transfer best across complexity?
6. **Cost-Performance**: What's the tradeoff between compute/API costs and performance?

### 4.2 Statistical Analysis (8 hrs)

**Significance Testing**:
```python
# analysis/statistical_tests.py
from scipy import stats
from statsmodels.stats.multitest import multipletests

def compare_all_agents(results):
    """Pairwise comparisons with Bonferroni correction"""
    agents = list(results.keys())
    n_comparisons = len(agents) * (len(agents) - 1) // 2

    p_values = []
    comparisons = []

    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            agent_a = results[agents[i]]
            agent_b = results[agents[j]]

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(agent_a, agent_b)
            p_values.append(p_val)
            comparisons.append((agents[i], agents[j], t_stat))

    # Bonferroni correction
    reject, p_corrected, _, _ = multipletests(p_values, method='bonferroni')

    return comparisons, p_corrected, reject

def compute_effect_size(group_a, group_b):
    """Cohen's d effect size"""
    mean_diff = np.mean(group_a) - np.mean(group_b)
    pooled_std = np.sqrt((np.std(group_a)**2 + np.std(group_b)**2) / 2)
    return mean_diff / pooled_std
```

**Confidence Intervals**:
```python
def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence intervals"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    return np.percentile(bootstrap_means, [alpha*100, (1-alpha)*100])
```

### 4.3 Visualization Suite (15 hrs)

**1. Performance vs Board Size** (3 hrs)
```python
# analysis/visualization.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_scaling_curves(results, agents):
    """Performance degradation with board size"""
    fig, ax = plt.subplots(figsize=(10, 6))

    board_sizes = [5, 6, 7, 8, 10, 12]

    for agent in agents:
        means = [results[agent][size]['mean'] for size in board_sizes]
        ci_lower = [results[agent][size]['ci_95'][0] for size in board_sizes]
        ci_upper = [results[agent][size]['ci_95'][1] for size in board_sizes]

        ax.plot(board_sizes, means, marker='o', label=agent)
        ax.fill_between(board_sizes, ci_lower, ci_upper, alpha=0.2)

    ax.set_xlabel('Board Size')
    ax.set_ylabel('Moves to Completion')
    ax.set_title('Agent Performance vs Board Size (with 95% CI)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
```

**2. Agent × Complexity Heatmap** (2 hrs)
```python
def plot_performance_heatmap(results):
    """Heatmap of agent performance across all conditions"""
    # Rows: agents, Columns: complexity conditions
    # Color: normalized performance (0=worst, 1=best)
    pass
```

**3. RL Learning Curves** (2 hrs)
```python
def plot_training_curves(tensorboard_logs):
    """Training progression with variance across seeds"""
    # Plot reward vs timesteps with shaded variance
    pass
```

**4. Strategy Visualization** (3 hrs)
```python
def plot_attack_heatmaps(game_trajectories, agents):
    """Where does each agent attack? Aggregate across games"""
    # 10×10 heatmap showing attack frequency per cell
    # Compare: Do RL agents learn to focus on center like probability agent?
    pass

def plot_learned_vs_optimal_probability(rl_agent, probability_agent):
    """Compare RL agent's targeting to optimal probability"""
    # Side-by-side heatmaps
    # Correlation analysis
    pass
```

**5. Trajectory Animations** (2 hrs)
```python
def create_decision_explanation_video(game, agent):
    """Step-through with agent's decision rationale"""
    # Frame-by-frame progression
    # Overlay: agent's reasoning, probability map, chosen action
    pass
```

**6. Cost-Performance Pareto Frontier** (2 hrs)
```python
def plot_cost_performance_tradeoff(agents):
    """Pareto frontier: performance vs total cost"""
    fig, ax = plt.subplots()

    for agent in agents:
        x = agent.total_cost  # Training + inference
        y = agent.performance  # Median moves
        ax.scatter(x, y, s=100, label=agent.name)

    ax.set_xlabel('Total Cost ($)')
    ax.set_ylabel('Performance (moves, lower=better)')
    ax.set_title('Cost-Performance Tradeoff')
    ax.legend()

    return fig
```

**7. Additional Visualizations** (1 hr)
- Distribution plots (box plots, violin plots)
- Win rate matrices (tournament results)
- Transfer learning heatmaps (train size → test size)

### 4.4 Qualitative Analysis (5 hrs)

**LLM Reasoning Categorization**:
- Read 50 LLM game transcripts
- Categorize reasoning patterns
- Identify failure modes
- Extract interesting strategies

**RL Strategy Emergence**:
- Visualize learned value functions
- Check if hunt/target mode emerges
- Analyze failure cases

**Deliverables**:
- ✅ Statistical test results (p-values, effect sizes, CIs)
- ✅ 15-20 publication-quality figures
- ✅ `analysis.ipynb` with all figures and analysis
- ✅ Qualitative findings document

---

## 🌐 Phase 5: Interactive Website (Weeks 7-8, ~40 hrs)

**Goal**: Free-tier hosted website for gameplay and analysis visualization.

### 5.1 Architecture (GitHub Pages + Netlify Functions) (10 hrs)

**Tech Stack**:
- **Frontend**: React/Next.js static site on GitHub Pages
- **Backend**: Netlify serverless functions (125K requests/month free)
- **Storage**: GitHub Gists (free, JSON storage) or Firebase free tier
- **Deployment**: GitHub Actions CI/CD

**Why This Stack**:
- $0 hosting cost
- 125K requests = ~4,000 games/month (30 moves/game)
- Static site = fast, SEO-friendly, easy to maintain
- Serverless = no server management

**Alternative (WASM)**:
- Compile Python agents to WebAssembly
- Fully client-side (no API calls)
- Slower but unlimited free usage

**Implementation**:
```
battleship-website/
├── frontend/              # React app
│   ├── pages/
│   │   ├── index.tsx     # Landing page
│   │   ├── play.tsx      # Interactive game
│   │   ├── analysis.tsx  # Dashboard
│   │   └── about.tsx     # Research writeup
│   ├── components/
│   │   ├── GameBoard.tsx
│   │   ├── AgentSelector.tsx
│   │   ├── ProbabilityHeatmap.tsx
│   │   └── StatsDisplay.tsx
│   └── public/
│       └── analysis_data/ # Static JSON exports
│
├── netlify/functions/     # Serverless backend
│   ├── get_agent_move.ts # Agent move calculation
│   ├── log_game.ts       # Store game results
│   └── get_stats.ts      # Aggregate statistics
│
└── deploy/
    └── github_action.yml # Auto-deploy
```

### 5.2 Features (25 hrs)

**5.2a: Gameplay Interface (12 hrs)**

```tsx
// frontend/pages/play.tsx
import { useState } from 'react';
import GameBoard from '../components/GameBoard';
import AgentSelector from '../components/AgentSelector';

export default function PlayPage() {
    const [opponent, setOpponent] = useState('probability');
    const [boardSize, setBoardSize] = useState(10);
    const [gameState, setGameState] = useState(null);

    const handleClick = async (row, col) => {
        // Player attacks cell
        const action = row * boardSize + col;

        // Get agent response
        const response = await fetch('/.netlify/functions/get_agent_move', {
            method: 'POST',
            body: JSON.stringify({ gameState, agent: opponent })
        });

        const agentMove = await response.json();

        // Update game state
        setGameState(agentMove.newState);
    };

    return (
        <div>
            <AgentSelector value={opponent} onChange={setOpponent} />
            <GameBoard state={gameState} onCellClick={handleClick} />
            <ProbabilityHeatmap agent={opponent} state={gameState} />
        </div>
    );
}
```

**Features**:
- Select opponent: Random, Probability, PPO, DQN, LLM
- Select difficulty: Board size (5×5 to 12×12), ship config
- Interactive grid with click-to-attack
- Real-time probability heatmap overlay (show agent's "thinking")
- Move history and running statistics
- Replay controls (step forward/back)

**5.2b: Analysis Dashboard (8 hrs)**

```tsx
// frontend/pages/analysis.tsx
import dynamic from 'next/dynamic';

// Lazy load Plotly (large bundle)
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function AnalysisPage() {
    // Load static JSON data generated from Phase 4
    const data = require('../public/analysis_data/results.json');

    return (
        <div>
            <h1>Research Findings</h1>

            {/* Interactive Plotly charts */}
            <Plot
                data={data.performance_curves}
                layout={{ title: 'Agent Performance vs Board Size' }}
            />

            <Plot
                data={data.cost_performance}
                layout={{ title: 'Cost-Performance Tradeoff' }}
            />

            {/* Filterable table */}
            <AgentComparisonTool data={data.agent_stats} />

            {/* Key findings */}
            <ResearchSummary findings={data.key_findings} />
        </div>
    );
}
```

**Features**:
- Embedded interactive Plotly figures from Phase 4
- Agent comparison tool: side-by-side stats
- Complexity explorer: filter by board size, config, rules
- Research findings summary (key takeaways)
- Download data: CSV/JSON exports for reproducibility

**5.2c: Data Collection (5 hrs)**

```typescript
// netlify/functions/log_game.ts
import { Handler } from '@netlify/functions';

export const handler: Handler = async (event) => {
    const game = JSON.parse(event.body);

    // Validate
    if (!game.opponent || !game.moves || !game.outcome) {
        return { statusCode: 400, body: 'Invalid data' };
    }

    // Store in GitHub Gist or Firebase
    await storeGameResult({
        timestamp: Date.now(),
        opponent: game.opponent,
        board_size: game.board_size,
        moves: game.moves,
        outcome: game.outcome,
        player_won: game.player_won,
        hit_rate: game.hit_rate,
    });

    return { statusCode: 200, body: 'Logged' };
};
```

**Privacy**:
- No PII collected
- Anonymous gameplay data only
- Opt-in analytics
- GDPR compliant

**Stats Collected**:
- Games played per agent
- Human win rate vs each agent
- Average moves, hit rate
- Popular board sizes

### 5.3 Portfolio Presentation (5 hrs)

**Landing Page**:
```
Hero Section:
  - "Battleship AI: Comparing RL, LLM, and Optimal Strategies"
  - "Play Now" CTA button
  - GIF: animated game demo

Overview:
  - 3-sentence project summary
  - Key findings (3 bullet points with numbers)
  - "Read Full Analysis" link

Quick Stats:
  - 7 agents compared
  - 54 complexity conditions
  - 1000+ games analyzed
  - Publication-quality results

Navigation:
  - Play Interactive Demo
  - Explore Analysis
  - Read Research Writeup
  - View Code (GitHub link)
```

**Design**:
- Clean, professional, mobile-responsive
- Fast load times (<2s)
- Accessible (WCAG AA)
- SEO optimized (meta tags, OpenGraph)

**Deliverables**:
- ✅ Live website on GitHub Pages (custom domain optional)
- ✅ 3-4 playable agents
- ✅ Interactive analysis dashboard
- ✅ Data collection infrastructure
- ✅ Professional portfolio presentation

---

## 📝 Phase 6: Technical Writeup (Parallel, ~25 hrs)

**Goal**: Conference-quality paper or high-impact blog post.

### 6.1 Structure (Blog Post Format, 3000-5000 words)

**I. Introduction** (500 words)
- Motivation: Why compare RL vs LLM for game-playing?
- Battleship as testbed: Discrete, strategic, tractable
- Research questions (list 5 key questions)
- Preview of findings

**II. Background** (700 words)
- Game rules and complexity
- Related work:
  - AlphaZero (RL for games)
  - lmgame-Bench (LLM for games)
  - DataGenetics algorithm (optimal baseline)
- Technical overview: Gymnasium, Stable-Baselines3, Claude API

**III. Methodology** (1000 words)
- Environment design
  - Gymnasium implementation
  - Observation/action spaces
  - Reward structure
- Agent implementations
  - RL: PPO, DQN, A2C, Rainbow (training details)
  - LLM: Claude, Llama, Mistral (prompting strategies)
  - Baseline: Probability agent
- Experimental design
  - Complexity dimensions (board size, ships, rules, opponents)
  - Evaluation metrics
  - Statistical testing
- Reproducibility
  - Seeds, versions, hyperparameters
  - Code/data release

**IV. Results** (1500 words)
- **RQ1: Scaling** - Performance vs board size (Figure 1)
  - Finding: RL degrades gracefully, LLM struggles on 12×12
- **RQ2: Sample Efficiency** - Training time vs zero-shot (Figure 2)
  - Finding: LLM wins <100 games, RL wins 1000+ games
- **RQ3: Strategy** - Learned vs optimal (Figure 3, 4)
  - Finding: PPO learns probability-like strategy, DQN doesn't
- **RQ4: Failure Modes** - Error analysis (Figure 5)
  - Finding: LLM spatial errors, RL explore-exploit tradeoff
- **RQ5: Generalization** - Transfer learning (Figure 6)
  - Finding: PPO transfers well, DQN needs retraining
- **RQ6: Cost-Performance** - Pareto frontier (Figure 7)
  - Finding: Probability agent best value, Claude best zero-shot

**V. Discussion** (800 words)
- Key insights
  - When to use RL vs LLM
  - Hybrid approaches (future work)
- Limitations
  - Single game domain
  - Limited LLM budget
  - No human baseline
- Implications for AI research
  - LLMs struggle with spatial reasoning
  - RL sample efficiency is real concern
  - Simple heuristics often sufficient
- Future directions
  - Hybrid RL+LLM agents
  - Multi-agent competition
  - Human studies

**VI. Conclusion** (300 words)
- Summary of findings
- Broader impact: decision framework for practitioners
- Code/data availability
- Call to action: "Play the demo!"

### 6.2 Reproducibility (5 hrs)

**Code Release**:
```
github.com/username/battleship-ai-research/
├── README.md              # Setup instructions
├── requirements.txt       # Exact versions
├── environment.yml        # Conda environment
├── game/                  # Environment code
├── training/              # RL training scripts
├── experiments/           # Evaluation scripts
├── analysis/              # Analysis notebooks
├── results/               # Saved results
├── models/                # Trained RL agents
└── docker/
    └── Dockerfile         # One-command reproducibility
```

**Documentation**:
```markdown
# Reproducibility Guide

## Quick Start
```bash
docker run -it battleship-ai:latest
python experiments/reproduce_all_results.py
```

## Manual Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download models: `python download_models.py`
3. Run evaluation: `python experiments/evaluate_all.py`
4. Generate figures: `jupyter notebook analysis/figures.ipynb`

## Results
All figures and tables reproduced in `output/reproduced/`
```

**Data Release**:
- Game trajectories (100 games per agent, JSON format)
- Trained RL models (checkpoints, < 100MB each)
- Evaluation results (CSV with all metrics)
- Analysis notebooks (Jupyter with outputs)

**Docker Container**:
```dockerfile
FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "experiments/reproduce_all_results.py"]
```

### 6.3 Writing & Editing (15 hrs)

**Process**:
1. First draft (8 hrs)
2. Figures integration (2 hrs)
3. Editing for clarity (3 hrs)
4. Proofreading (2 hrs)

**Style**:
- Academic but accessible
- Active voice
- Concrete numbers over vague claims
- Figures tell the story

**Deliverables**:
- ✅ 3000-5000 word blog post (Markdown)
- ✅ Optional: LaTeX paper for arXiv
- ✅ GitHub repo with reproducibility badge
- ✅ All code, data, models released

---

## 🎯 Success Metrics

### Portfolio Impact
- ✅ Live interactive demo (portfolio centerpiece)
- ✅ Rigorous methodology (demonstrates research capability)
- ✅ Clear communication (writeup + visualizations)
- ✅ Full reproducibility (professional engineering)

### Research Quality
- ✅ Novel insights (RL strategy emergence, LLM reasoning patterns)
- ✅ Statistical rigor (CIs, p-values, effect sizes on all claims)
- ✅ Comprehensive evaluation (500+ agent-condition pairs)
- ✅ Publication-ready figures and writing

### Budget Adherence
- ✅ API costs: $50-100
  - Open-source LLMs: $0
  - Claude API: ~$50-100 with prompt caching
  - GPT-4o (optional): if budget allows
- ✅ Compute: <$50
  - Colab Pro: $10/month × 2 months = $20
  - Lambda Labs: ~$0.50/hr × 40 hrs = $20
  - CPU training: $0
- ✅ Hosting: $0
  - GitHub Pages: free
  - Netlify functions: free tier (125K requests/month)
- **Total**: ~$100-150 all-in

---

## 🚀 Optional Extensions (If Time Permits)

### If Ahead of Schedule

1. **Multi-Agent RL** (2 weeks)
   - Self-play training
   - Population-based training
   - Emergent strategies

2. **Hybrid Agents** (2 weeks)
   - LLM generates candidates, RL selects
   - Division of cognitive labor
   - Performance vs interpretability

3. **Interpretability** (1 week)
   - Attention visualization
   - Saliency maps
   - Decision trees from RL policies

4. **Human Study** (2 weeks)
   - Recruit players via website
   - Human vs AI comparison
   - Strategy analysis

5. **Tournament System** (1 week)
   - Round-robin all agents
   - ELO rankings
   - Head-to-head visualizations

### If Behind Schedule

**Cut in priority order**:
1. Rainbow DQN (keep PPO, DQN, A2C)
2. Advanced rule variants (keep standard only)
3. Fine-tuning experiments (LLM prompt engineering only)
4. Hybrid agents (future work)
5. Curriculum learning (simple transfer only)

**Core must-haves**:
- PPO + DQN
- Claude + 1 open-source LLM
- Probability baseline
- 3 board sizes (7×7, 10×10, 12×12)
- Basic website
- Statistical analysis with CIs

---

## 📦 Final Deliverables Summary

1. **Code Repository**
   - Clean, documented, tested
   - RL training scripts
   - LLM agent implementations
   - Evaluation framework
   - Analysis notebooks

2. **Trained Models**
   - 4 RL agents (PPO, DQN, A2C, Rainbow)
   - Checkpoints across training
   - Transfer learning models

3. **LLM Agents**
   - 3+ open-source LLM agents
   - Claude API agent with prompt caching
   - Prompt templates and variations

4. **Analysis Artifacts**
   - Jupyter notebook with all figures
   - 15-20 publication-quality figures
   - Statistical test results
   - Qualitative findings document

5. **Interactive Website**
   - Live demo on GitHub Pages
   - 3-4 playable agents
   - Analysis dashboard
   - Data collection system

6. **Technical Writeup**
   - 3000-5000 word blog post
   - Embedded figures and results
   - Code/data availability section
   - Optional LaTeX paper

7. **Data Release**
   - Game trajectories (100+ games per agent)
   - Evaluation results (CSV)
   - Trained models (< 500MB total)
   - Reproducibility scripts

---

## 💡 Key Principles

**Rigor Over Breadth**
- Better to do 3 agents perfectly than 10 poorly
- Every claim needs statistical backing
- Reproducibility is non-negotiable

**Budget Consciousness**
- Open-source first, API second
- Prompt caching for 85% cost reduction
- Free hosting, smart architecture

**Portfolio Focus**
- Live demo > static writeup
- Clear communication > technical jargon
- Professional presentation matters

**Time Management**
- Infrastructure first (Phase 1)
- Parallel work where possible (writeup during training)
- Cut optional features, not rigor

---

## 📚 References & Resources

### Key Papers
- DataGenetics Battleship: http://www.datagenetics.com/blog/december32011/
- lmgame-Bench 2025: LLM game-playing evaluation
- Language Agents with RL for Strategic Play
- AlphaZero (Silver et al.)

### Tools Documentation
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- Weights & Biases: https://docs.wandb.ai/
- Anthropic API: https://docs.anthropic.com/
- Netlify Functions: https://docs.netlify.com/functions/

### Code Examples
- `game/env.py` - Current Gymnasium environment
- `game/agents/probability_agent.py` - Optimal baseline
- `demo_probability.py` - Example usage

---

**This plan prioritizes rigor, reproducibility, and portfolio impact. Execute methodically, track everything, and ship a project that demonstrates world-class research capability.**


Gemini prompt:
So i am training an agent to play the game of battleship. There is no adversary. The goal is to make the agent learn the optimal strategy -- the agent has two modes of operation:
1. target/exploitation mode: 
a) if for a move, there is a hit on a ship, the next move should be on a cell adjacent to the one just hit. 
b) Added complexity: if the ship did not sink, then continue in a straight path on either directions till the ship sinks. 
2. parity/exploration mode: 
a) For a move, if there is a miss, do not choose the adjacent cells, but the cell one over the adjacent cells. this give more coverage. 
b) Added complexity: if the length of the ship is 2, then alternate parity exploration is optimal, but if it is 3, the agent should skip 2 cells after a miss to explore further. 

I am using PPO. I am able to achieve 1.a with and without reward shaping. I am not able to achieve 2.a without reward shaping -- but with reward shaping, i can. 1.b and 2b are too complicated right now. What would be my best bet to make the agent learn 2a without reward shaping, and as an emergent behaviour? what about 1.b, 2b?