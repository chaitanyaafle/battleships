"""Probability heatmap renderer for DataGenetics strategy visualization."""

from game.state import GameState
from game.agents.probability_agent import ProbabilityAgent
import numpy as np
from typing import Optional, Tuple


def render_probability_html(
    state: GameState,
    agent: ProbabilityAgent,
    next_action: Optional[int] = None
) -> str:
    """
    Render game state with probability density heatmap.

    Shows side-by-side visualization of:
    - Attack board (current state)
    - Probability density heatmap

    Args:
        state: Current game state
        agent: ProbabilityAgent instance (to access probability grid)
        next_action: Optional next action to highlight

    Returns:
        HTML string with side-by-side visualization
    """
    rows, cols = state.board_size
    probability_grid = agent.get_probability_grid()

    if probability_grid is None:
        # If no probability grid available, fall back to basic render
        from game.renderers.html import render_html
        return render_html(state)

    # Normalize probabilities for color mapping (0-1 range)
    max_prob = np.max(probability_grid)
    min_prob = np.min(probability_grid)

    if max_prob > min_prob:
        normalized_prob = (probability_grid - min_prob) / (max_prob - min_prob)
    else:
        normalized_prob = np.zeros_like(probability_grid)

    # Calculate next action coordinates if provided
    next_row, next_col = None, None
    if next_action is not None:
        next_row, next_col = divmod(next_action, cols)

    # Build HTML
    html = _build_header(state)
    html += _build_side_by_side_boards(
        state, probability_grid, normalized_prob,
        next_row, next_col
    )
    html += _build_legend()
    html += _build_ship_status(state)
    html += _build_footer()

    return html


def _build_header(state: GameState) -> str:
    """Build HTML header section."""
    ships_remaining = len([s for s in state.ships.values() if not s.is_sunk])
    status = "Game Over - Victory!" if state.done else "In Progress"
    status_color = "#27ae60" if state.done else "#34495e"

    return f"""
    <div style="background-color: white; padding: 20px; font-family: Arial, sans-serif; max-width: 1400px;">
        <h2 style="color: #2c3e50;">DataGenetics Probability Strategy Visualization</h2>
        <div style="margin-bottom: 15px; background-color: #ecf0f1; padding: 10px; border-radius: 5px;">
            <strong>Move Count:</strong> {state.move_count} &nbsp;&nbsp;
            <strong>Ships Remaining:</strong> {ships_remaining} / {len(state.ships)} &nbsp;&nbsp;
            <strong>Status:</strong> <span style="color: {status_color};">{status}</span>
        </div>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
    """


def _build_side_by_side_boards(
    state: GameState,
    probability_grid: np.ndarray,
    normalized_prob: np.ndarray,
    next_row: Optional[int],
    next_col: Optional[int]
) -> str:
    """Build side-by-side attack board and probability heatmap."""
    rows, cols = state.board_size

    html = """
        <div style="flex: 1; min-width: 400px;">
            <h3 style="color: #2c3e50; text-align: center;">Attack Board</h3>
    """
    html += _build_attack_board_table(state, next_row, next_col)
    html += "</div>"

    html += """
        <div style="flex: 1; min-width: 400px;">
            <h3 style="color: #2c3e50; text-align: center;">Probability Density</h3>
    """
    html += _build_probability_heatmap_table(
        state.board_size, probability_grid, normalized_prob, next_row, next_col
    )
    html += "</div>"
    html += "</div>"

    return html


def _build_attack_board_table(
    state: GameState,
    next_row: Optional[int],
    next_col: Optional[int]
) -> str:
    """Build attack board visualization table."""
    rows, cols = state.board_size

    html = '<table style="border-collapse: collapse; border: 2px solid #2c3e50; margin: 0 auto;">'

    # Header row
    html += '<tr><th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;"></th>'
    for j in range(cols):
        html += f'<th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;">{j}</th>'
    html += '</tr>'

    # Data rows
    for i in range(rows):
        html += f'<tr><th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;">{chr(65+i)}</th>'
        for j in range(cols):
            color = _get_attack_cell_color(state, i, j)

            # Highlight next action with thick border
            border_style = "3px solid #f39c12" if (i == next_row and j == next_col) else "1px solid #2c3e50"

            html += f'<td style="border: {border_style}; background-color: {color}; width: 35px; height: 35px;"></td>'
        html += '</tr>'

    html += '</table>'
    return html


def _build_probability_heatmap_table(
    board_size: Tuple[int, int],
    probability_grid: np.ndarray,
    normalized_prob: np.ndarray,
    next_row: Optional[int],
    next_col: Optional[int]
) -> str:
    """Build probability heatmap visualization table."""
    rows, cols = board_size

    html = '<table style="border-collapse: collapse; border: 2px solid #2c3e50; margin: 0 auto;">'

    # Header row
    html += '<tr><th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;"></th>'
    for j in range(cols):
        html += f'<th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;">{j}</th>'
    html += '</tr>'

    # Data rows
    for i in range(rows):
        html += f'<tr><th style="border: 1px solid #2c3e50; padding: 8px; background-color: #ecf0f1;">{chr(65+i)}</th>'
        for j in range(cols):
            prob_value = probability_grid[i, j]
            norm_value = normalized_prob[i, j]

            # Color gradient: blue (low) -> green -> yellow -> red (high)
            color = _get_heatmap_color(norm_value)

            # Highlight next action with thick border
            border_style = "3px solid #f39c12" if (i == next_row and j == next_col) else "1px solid #2c3e50"

            # Add probability value as title (hover text)
            html += f'<td style="border: {border_style}; background-color: {color}; width: 35px; height: 35px; text-align: center; font-size: 10px; color: {"white" if norm_value > 0.5 else "#2c3e50"};" title="Probability: {prob_value:.1f}">'

            # Show value if it's the max
            if norm_value == 1.0 and prob_value > 0:
                html += f'<strong>{int(prob_value)}</strong>'

            html += '</td>'
        html += '</tr>'

    html += '</table>'
    return html


def _get_attack_cell_color(state: GameState, row: int, col: int) -> str:
    """Get color for attack board cell."""
    attack_val = state.attack_board[row, col]

    if attack_val == 0:  # Unknown
        return '#3498db'
    elif attack_val == 1:  # Miss
        return '#95a5a6'
    elif attack_val == 2:  # Hit
        # Check if ship is sunk
        for ship in state.ships.values():
            if (row, col) in ship.coords and ship.is_sunk:
                return '#2c3e50'  # Sunk
        return '#e74c3c'  # Regular hit

    return '#3498db'  # Default


def _get_heatmap_color(normalized_value: float) -> str:
    """
    Get color for heatmap cell based on normalized probability (0-1).

    Color gradient:
    - 0.0: Dark blue (very low probability)
    - 0.25: Light blue
    - 0.5: Yellow
    - 0.75: Orange
    - 1.0: Red (highest probability)
    """
    if normalized_value == 0:
        return '#f0f0f0'  # Light gray for zero probability

    # Use a blue -> cyan -> green -> yellow -> red gradient
    if normalized_value < 0.25:
        # Blue to cyan
        ratio = normalized_value * 4
        r = int(0 * (1 - ratio) + 100 * ratio)
        g = int(100 * (1 - ratio) + 200 * ratio)
        b = int(200 * (1 - ratio) + 255 * ratio)
    elif normalized_value < 0.5:
        # Cyan to green
        ratio = (normalized_value - 0.25) * 4
        r = int(100 * (1 - ratio) + 100 * ratio)
        g = int(200 * (1 - ratio) + 255 * ratio)
        b = int(255 * (1 - ratio) + 100 * ratio)
    elif normalized_value < 0.75:
        # Green to yellow
        ratio = (normalized_value - 0.5) * 4
        r = int(100 * (1 - ratio) + 255 * ratio)
        g = int(255 * (1 - ratio) + 255 * ratio)
        b = int(100 * (1 - ratio) + 0 * ratio)
    else:
        # Yellow to red
        ratio = (normalized_value - 0.75) * 4
        r = 255
        g = int(255 * (1 - ratio) + 100 * ratio)
        b = 0

    return f'#{r:02x}{g:02x}{b:02x}'


def _build_legend() -> str:
    """Build legend for both boards."""
    return """
        <div style="margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px;">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
                <div>
                    <strong>Attack Board Legend:</strong><br>
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #3498db; border: 1px solid black;"></span> Unknown &nbsp;
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #95a5a6; border: 1px solid black;"></span> Miss &nbsp;
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #e74c3c; border: 1px solid black;"></span> Hit &nbsp;
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #2c3e50; border: 1px solid black;"></span> Sunk &nbsp;
                    <span style="display: inline-block; width: 15px; height: 15px; border: 3px solid #f39c12; background-color: white;"></span> Next
                </div>
                <div>
                    <strong>Probability Heatmap:</strong><br>
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #64c8ff; border: 1px solid black;"></span> Low &nbsp;
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #64ff64; border: 1px solid black;"></span> Medium &nbsp;
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #ffff00; border: 1px solid black;"></span> High &nbsp;
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #ff6464; border: 1px solid black;"></span> Highest
                </div>
            </div>
        </div>
    """


def _build_ship_status(state: GameState) -> str:
    """Build ship status section."""
    html = '<div style="margin-top: 15px; font-size: 14px;"><strong>Ships:</strong><br>'

    for ship in state.ships.values():
        status_icon = "✓" if ship.is_sunk else "○"
        status_color = "#27ae60" if ship.is_sunk else "#e67e22"
        hit_info = f"{len(ship.hits)}/{ship.size}" if not ship.is_sunk else "SUNK"

        html += f'<div style="color: {status_color}; margin: 5px 0;">'
        html += f'{status_icon} <strong>{ship.name.title()}</strong> (Size {ship.size}): {hit_info}'
        html += '</div>'

    html += '</div>'
    return html


def _build_footer() -> str:
    """Build footer section."""
    return """
        <div style="margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #3498db; font-size: 12px; color: #555;">
            <strong>DataGenetics Optimal Strategy:</strong> The probability heatmap shows the likelihood
            of each cell containing a ship based on all possible valid placements of remaining ships.
            Higher values (red) indicate cells more likely to contain ships. The agent selects the
            cell with the highest probability each turn.
        </div>
    </div>
    """
