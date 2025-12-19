"""Quick test of LLM agent performance."""

from game.env import BattleshipEnv
from game.agents.llm_agent import LLMAgent
from game.agents.random_agent import RandomAgent
from game.agents.probability_agent import ProbabilityAgent
from training.evaluate import evaluate_agent


def main():
    # Test on small board first
    env = BattleshipEnv(board_size=(5, 5), verbose=False)

    print("="*80)
    print("TESTING LLM AGENT - ZERO-SHOT PERFORMANCE")
    print("="*80)

    # Baseline agents for comparison
    print("\n1. Testing Random Agent (baseline)...")
    random_agent = RandomAgent("Random")
    random_results = evaluate_agent(random_agent, env, n_episodes=10, seed=42)
    print(f"   Mean moves: {random_results['mean_length']:.1f}")
    print(f"   Median moves: {random_results['median_length']:.1f}")
    print(f"   Win rate: {random_results['win_rate']:.1%}")

    print("\n2. Testing Probability Agent (expert baseline)...")
    prob_agent = ProbabilityAgent("Probability")
    prob_results = evaluate_agent(prob_agent, env, n_episodes=10, seed=42)
    print(f"   Mean moves: {prob_results['mean_length']:.1f}")
    print(f"   Median moves: {prob_results['median_length']:.1f}")
    print(f"   Win rate: {prob_results['win_rate']:.1%}")

    # Test LLM agent (start with just one model)
    print("\n3. Testing LLM Agent (Llama-3.2-3B-Instruct)...")
    print("   This will download ~6GB if model not cached...")

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # model_name = "Qwen/Qwen2-1.5B-Instruct"

    try:
        agent = LLMAgent(
            name=f"LLM-Zero-Shot",
            model_name=model_name,
            max_new_tokens=10,
            device="cpu",  # Use CPU for stability (MPS has compatibility issues)
            temperature=0.1,  # Low temp for more deterministic outputs
        )

        results = evaluate_agent(agent, env, n_episodes=3, seed=42)

        print(f"\n   ✓ LLM Agent Results:")
        print(f"     Mean moves: {results['mean_length']:.1f}")
        print(f"     Median moves: {results['median_length']:.1f}")
        print(f"     Win rate: {results['win_rate']:.1%}")

        # Compare to baselines
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"{'Agent':<30} {'Mean':>10} {'Median':>10} {'Win%':>10}")
        print(f"{'-'*80}")
        print(f"{'Random':<30} {random_results['mean_length']:>10.1f} {random_results['median_length']:>10.1f} {random_results['win_rate']*100:>10.1f}%")
        print(f"{'Probability (Expert)':<30} {prob_results['mean_length']:>10.1f} {prob_results['median_length']:>10.1f} {prob_results['win_rate']*100:>10.1f}%")
        print(f"{'LLM (Zero-shot)':<30} {results['mean_length']:>10.1f} {results['median_length']:>10.1f} {results['win_rate']*100:>10.1f}%")
        print(f"{'='*80}")

        # Analysis
        if results['mean_length'] < random_results['mean_length']:
            print("\n✓ LLM beats random! Zero-shot reasoning is working.")
        if results['mean_length'] < prob_results['mean_length']:
            print("✓ LLM beats expert! Impressive zero-shot performance.")
        else:
            improvement = ((random_results['mean_length'] - results['mean_length']) /
                          (random_results['mean_length'] - prob_results['mean_length']) * 100)
            print(f"\n✓ LLM is {improvement:.1f}% of the way from random to expert")
            print("   Fine-tuning should close this gap!")

    except Exception as e:
        print(f"\n✗ Error loading/testing LLM: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure transformers, torch are installed")
        print("  - Try device='cpu' if MPS fails")
        print("  - Check HuggingFace token if model requires authentication")


if __name__ == "__main__":
    main()
