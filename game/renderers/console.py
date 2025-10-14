"""Console renderer for Battleship game visualization."""

from game.state import GameState


def render_console(state: GameState) -> str:
    """
    Render game state as ASCII text for console.

    Symbols:
    - · : Unknown (not attacked)
    - ○ : Miss
    - ✕ : Hit
    - ■ : Sunk ship

    Args:
        state: Current game state

    Returns:
        ASCII string representation of the game
    """
    rows, cols = state.board_size

    # Determine cell symbol
    def get_cell_symbol(r, c):
        attack_val = state.attack_board[r, c]
        if attack_val == 0:  # Unknown
            return '·'
        elif attack_val == 1:  # Miss
            return '○'
        elif attack_val == 2:  # Hit
            # Check if ship is sunk
            for ship in state.ships.values():
                if (r, c) in ship.coords and ship.is_sunk:
                    return '■'  # Sunk
            return '✕'  # Regular hit
        return '·'  # Default

    # Build ASCII output
    lines = []
    lines.append("=" * (cols * 2 + 5))
    lines.append(f"Battleship Game - Move {state.move_count}")
    lines.append(f"Ships: {len([s for s in state.ships.values() if not s.is_sunk])} / {len(state.ships)}")
    if state.done:
        lines.append("Status: VICTORY!")
    lines.append("=" * (cols * 2 + 5))
    lines.append("")

    # Column headers
    header = "   " + " ".join(f"{i:>1}" for i in range(cols))
    lines.append(header)

    # Grid rows
    for i in range(rows):
        row_label = chr(65 + i)
        row = f"{row_label}  " + " ".join(get_cell_symbol(i, j) for j in range(cols))
        lines.append(row)

    lines.append("")
    lines.append("Legend: · = Unknown, ○ = Miss, ✕ = Hit, ■ = Sunk")
    lines.append("")

    # Ship status
    lines.append("Ships:")
    for ship in state.ships.values():
        status = "[SUNK]" if ship.is_sunk else f"[{len(ship.hits)}/{ship.size}]"
        lines.append(f"  {ship.name.title()}: {status}")

    lines.append("=" * (cols * 2 + 5))

    return "\n".join(lines)
