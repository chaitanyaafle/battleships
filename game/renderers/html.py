"""HTML renderer for Battleship game visualization."""

from game.state import GameState


def render_html(state: GameState) -> str:
    """
    Render game state as HTML.

    Color scheme:
    - Blue (#3498db): Unknown cells
    - Gray (#95a5a6): Miss
    - Red (#e74c3c): Hit
    - Black (#2c3e50): Sunk ship

    Args:
        state: Current game state

    Returns:
        HTML string representation of the game
    """
    rows, cols = state.board_size

    # Determine cell color
    def get_cell_color(r, c):
        attack_val = state.attack_board[r, c]
        if attack_val == 0:  # Unknown
            return '#3498db'
        elif attack_val == 1:  # Miss
            return '#95a5a6'
        elif attack_val == 2:  # Hit
            # Check if ship is sunk
            for ship in state.ships.values():
                if (r, c) in ship.coords and ship.is_sunk:
                    return '#2c3e50'  # Sunk
            return '#e74c3c'  # Regular hit
        return '#3498db'  # Default

    # Build HTML
    html = """
    <div style="background-color: white; padding: 20px; font-family: Arial, sans-serif;">
        <h2 style="color: #2c3e50;">Battleship Game</h2>
        <div style="margin-bottom: 15px;">
            <strong>Move Count:</strong> {move_count}<br>
            <strong>Ships Remaining:</strong> {ships_remaining} / {total_ships}<br>
            <strong>Status:</strong> <span style="color: {status_color};">{status}</span>
        </div>
        <table style="border-collapse: collapse; border: 2px solid #2c3e50;">
            <tr>
                <th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;"></th>
    """.format(
        move_count=state.move_count,
        ships_remaining=len([s for s in state.ships.values() if not s.is_sunk]),
        total_ships=len(state.ships),
        status="Game Over - Victory!" if state.done else "In Progress",
        status_color="#27ae60" if state.done else "#34495e"
    )

    # Column headers
    for j in range(cols):
        html += f'<th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;">{j}</th>'
    html += '</tr>\n'

    # Grid rows
    for i in range(rows):
        html += f'<tr><th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;">{chr(65+i)}</th>'
        for j in range(cols):
            color = get_cell_color(i, j)
            html += f'<td style="border: 1px solid #2c3e50; background-color: {color}; width: 30px; height: 30px;"></td>'
        html += '</tr>\n'

    html += """
        </table>
        <div style="margin-top: 15px;">
            <strong>Legend:</strong><br>
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #3498db; border: 1px solid black;"></span> Unknown &nbsp;
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #95a5a6; border: 1px solid black;"></span> Miss &nbsp;
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #e74c3c; border: 1px solid black;"></span> Hit &nbsp;
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #2c3e50; border: 1px solid black;"></span> Sunk
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #7f8c8d;">
            <strong>Ships:</strong>
    """

    # Add ship status
    for ship in state.ships.values():
        status_icon = "✓" if ship.is_sunk else "○"
        status_color = "#27ae60" if ship.is_sunk else "#e67e22"
        html += f'<span style="color: {status_color}; margin-right: 10px;">{status_icon} {ship.name.title()} ({ship.size})</span>'

    html += """
        </div>
    </div>
    """

    return html
