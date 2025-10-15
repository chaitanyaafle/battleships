"""
Create animated HTML demo of probability agent gameplay.

This script generates a single interactive HTML file showing
the complete game progression with step-by-step visualization
of the probability density calculations.
"""

from game.env import BattleshipEnv
from game.agents.probability_agent import ProbabilityAgent
from game.renderers.animated_html import record_game_snapshots, create_animated_html


def main():
    """Generate animated HTML demo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate animated HTML demo of probability agent"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for game'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='battleship_animated_demo.html',
        help='Output HTML file path'
    )
    parser.add_argument(
        '--board-size',
        type=int,
        nargs=2,
        default=[10, 10],
        metavar=('ROWS', 'COLS'),
        help='Board size (default: 10 10)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Generating Animated HTML Demo")
    print("=" * 70)
    print(f"Seed: {args.seed}")
    print(f"Board size: {args.board_size[0]}x{args.board_size[1]}")
    print()

    # Create environment and agent
    env = BattleshipEnv(board_size=tuple(args.board_size), render_mode="html")
    agent = ProbabilityAgent(hit_adjacency_weight=50.0)

    print("Recording game progression...")
    snapshots, stats = record_game_snapshots(env, agent, seed=args.seed)

    print(f"Game completed in {stats['moves']} moves")
    print(f"Total snapshots: {len(snapshots)}")
    print()

    print("Generating HTML...")
    html = create_animated_html(
        snapshots=snapshots,
        board_size=tuple(args.board_size),
        total_ships=stats['total_ships'],
        final_moves=stats['moves']
    )

    # Save to file
    with open(args.output, 'w') as f:
        f.write(html)

    print(f"✓ Animated demo saved to: {args.output}")
    print()
    print("Open the file in a web browser to view the interactive demo.")
    print("Use the controls to step through the game move-by-move!")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
