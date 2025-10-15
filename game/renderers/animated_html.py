"""Animated HTML renderer for game progression visualization."""

import json
from typing import List, Tuple
import numpy as np
from game.state import GameState
from game.agents.probability_agent import ProbabilityAgent


class GameSnapshot:
    """Snapshot of game state at a specific move."""

    def __init__(
        self,
        move_number: int,
        attack_board: np.ndarray,
        probability_grid: np.ndarray,
        action: int,
        result: str,
        ship_sunk: str = None,
        ships_remaining: int = 0
    ):
        """Initialize game snapshot."""
        self.move_number = move_number
        self.attack_board = attack_board.copy()
        self.probability_grid = probability_grid.copy() if probability_grid is not None else None
        self.action = action
        self.result = result
        self.ship_sunk = ship_sunk
        self.ships_remaining = ships_remaining


def create_animated_html(
    snapshots: List[GameSnapshot],
    board_size: Tuple[int, int],
    total_ships: int,
    final_moves: int
) -> str:
    """
    Create animated HTML visualization of game progression.

    Args:
        snapshots: List of GameSnapshot objects
        board_size: (rows, cols) tuple
        total_ships: Total number of ships
        final_moves: Total moves to complete game

    Returns:
        HTML string with embedded JavaScript for animation
    """
    rows, cols = board_size

    # Convert snapshots to JSON-serializable format
    snapshots_data = []
    for snap in snapshots:
        # Normalize probabilities
        max_prob = np.max(snap.probability_grid) if snap.probability_grid is not None else 1
        min_prob = np.min(snap.probability_grid) if snap.probability_grid is not None else 0

        if max_prob > min_prob and snap.probability_grid is not None:
            normalized = ((snap.probability_grid - min_prob) / (max_prob - min_prob)).tolist()
        else:
            normalized = np.zeros_like(snap.probability_grid).tolist() if snap.probability_grid is not None else None

        snapshots_data.append({
            'move': snap.move_number,
            'attack_board': snap.attack_board.tolist(),
            'probability_grid': snap.probability_grid.tolist() if snap.probability_grid is not None else None,
            'normalized_prob': normalized,
            'action': snap.action,
            'result': snap.result,
            'ship_sunk': snap.ship_sunk,
            'ships_remaining': snap.ships_remaining
        })

    html = _build_html_template(
        snapshots_data=json.dumps(snapshots_data),
        rows=rows,
        cols=cols,
        total_ships=total_ships,
        final_moves=final_moves
    )

    return html


def _build_html_template(
    snapshots_data: str,
    rows: int,
    cols: int,
    total_ships: int,
    final_moves: int
) -> str:
    """Build complete HTML template with embedded JavaScript."""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Battleship Game Progression - DataGenetics Strategy</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            margin: 0;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #2c3e50;
            text-align: center;
        }}

        .controls {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }}

        .controls button {{
            margin: 0 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}

        .controls button:hover {{
            background-color: #2980b9;
        }}

        .controls button:disabled {{
            background-color: #95a5a6;
            cursor: not-allowed;
        }}

        .info-panel {{
            margin: 15px 0;
            padding: 15px;
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            border-radius: 5px;
        }}

        .boards-container {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }}

        .board-wrapper {{
            flex: 1;
            min-width: 400px;
        }}

        .board-wrapper h2 {{
            color: #2c3e50;
            text-align: center;
        }}

        table {{
            border-collapse: collapse;
            border: 2px solid #2c3e50;
            margin: 0 auto;
        }}

        td, th {{
            border: 1px solid #2c3e50;
            width: 35px;
            height: 35px;
            text-align: center;
            font-size: 12px;
        }}

        th {{
            background-color: #ecf0f1;
            padding: 8px;
        }}

        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            text-align: center;
        }}

        .color-sample {{
            display: inline-block;
            width: 15px;
            height: 15px;
            border: 1px solid black;
            margin: 0 3px;
        }}

        .slider-container {{
            margin: 15px 0;
        }}

        #moveSlider {{
            width: 80%;
            margin: 10px auto;
        }}

        .speed-control {{
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Battleship Game Progression</h1>
        <h3 style="text-align: center; color: #7f8c8d;">DataGenetics Probability Density Strategy</h3>

        <div class="info-panel">
            <strong>Game Statistics:</strong><br>
            Move: <span id="currentMove">0</span> / {final_moves} &nbsp;|&nbsp;
            Ships Remaining: <span id="shipsRemaining">{total_ships}</span> / {total_ships} &nbsp;|&nbsp;
            Last Result: <span id="lastResult">-</span>
        </div>

        <div class="controls">
            <button id="firstBtn" onclick="goToFirst()">⏮ First</button>
            <button id="prevBtn" onclick="previousMove()">◀ Previous</button>
            <button id="playBtn" onclick="togglePlay()">▶ Play</button>
            <button id="nextBtn" onclick="nextMove()">Next ▶</button>
            <button id="lastBtn" onclick="goToLast()">Last ⏭</button>
        </div>

        <div class="slider-container">
            <input type="range" id="moveSlider" min="0" max="{len(json.loads(snapshots_data)) - 1}" value="0" oninput="sliderChanged(this.value)">
        </div>

        <div class="speed-control">
            <label for="speedSlider">Animation Speed:</label>
            <input type="range" id="speedSlider" min="100" max="2000" value="1000" step="100">
            <span id="speedLabel">1.0s</span>
        </div>

        <div class="boards-container">
            <div class="board-wrapper">
                <h2>Attack Board</h2>
                <table id="attackBoard"></table>
            </div>
            <div class="board-wrapper">
                <h2>Probability Density</h2>
                <table id="probabilityBoard"></table>
            </div>
        </div>

        <div class="legend">
            <div style="margin: 10px 0;">
                <strong>Attack Board:</strong>
                <span class="color-sample" style="background-color: #3498db;"></span> Unknown
                <span class="color-sample" style="background-color: #95a5a6;"></span> Miss
                <span class="color-sample" style="background-color: #e74c3c;"></span> Hit
                <span class="color-sample" style="background-color: #2c3e50;"></span> Sunk
            </div>
            <div style="margin: 10px 0;">
                <strong>Probability:</strong>
                <span class="color-sample" style="background-color: #64c8ff;"></span> Low
                <span class="color-sample" style="background-color: #64ff64;"></span> Medium
                <span class="color-sample" style="background-color: #ffff00;"></span> High
                <span class="color-sample" style="background-color: #ff6464;"></span> Highest
            </div>
        </div>

        <div style="margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #3498db; font-size: 12px; color: #555;">
            <strong>About this visualization:</strong> This shows how the DataGenetics optimal strategy
            plays Battleship. The probability heatmap updates after each move, showing where remaining
            ships are most likely located based on all valid placements.
        </div>
    </div>

    <script>
        const snapshots = {snapshots_data};
        const rows = {rows};
        const cols = {cols};
        let currentIndex = 0;
        let isPlaying = false;
        let playInterval = null;

        // Initialize boards
        function initBoards() {{
            const attackBoard = document.getElementById('attackBoard');
            const probBoard = document.getElementById('probabilityBoard');

            // Create header row
            let headerHtml = '<tr><th></th>';
            for (let j = 0; j < cols; j++) {{
                headerHtml += `<th>${{j}}</th>`;
            }}
            headerHtml += '</tr>';

            // Create grid rows
            let gridHtml = '';
            for (let i = 0; i < rows; i++) {{
                gridHtml += `<tr><th>${{String.fromCharCode(65 + i)}}</th>`;
                for (let j = 0; j < cols; j++) {{
                    gridHtml += `<td id="attack_${{i}}_${{j}}"></td>`;
                }}
                gridHtml += '</tr>';
            }}

            attackBoard.innerHTML = headerHtml + gridHtml;

            // Same for probability board
            gridHtml = '';
            for (let i = 0; i < rows; i++) {{
                gridHtml += `<tr><th>${{String.fromCharCode(65 + i)}}</th>`;
                for (let j = 0; j < cols; j++) {{
                    gridHtml += `<td id="prob_${{i}}_${{j}}"></td>`;
                }}
                gridHtml += '</tr>';
            }}

            probBoard.innerHTML = headerHtml + gridHtml;
        }}

        function getAttackCellColor(value) {{
            if (value === 0) return '#3498db';  // Unknown
            if (value === 1) return '#95a5a6';  // Miss
            if (value === 2) return '#e74c3c';  // Hit (simplified)
            return '#3498db';
        }}

        function getHeatmapColor(normalized) {{
            if (normalized === 0) return '#f0f0f0';

            let r, g, b;
            if (normalized < 0.25) {{
                const ratio = normalized * 4;
                r = Math.floor(0 * (1 - ratio) + 100 * ratio);
                g = Math.floor(100 * (1 - ratio) + 200 * ratio);
                b = Math.floor(200 * (1 - ratio) + 255 * ratio);
            }} else if (normalized < 0.5) {{
                const ratio = (normalized - 0.25) * 4;
                r = Math.floor(100 * (1 - ratio) + 100 * ratio);
                g = Math.floor(200 * (1 - ratio) + 255 * ratio);
                b = Math.floor(255 * (1 - ratio) + 100 * ratio);
            }} else if (normalized < 0.75) {{
                const ratio = (normalized - 0.5) * 4;
                r = Math.floor(100 * (1 - ratio) + 255 * ratio);
                g = Math.floor(255 * (1 - ratio) + 255 * ratio);
                b = Math.floor(100 * (1 - ratio) + 0 * ratio);
            }} else {{
                const ratio = (normalized - 0.75) * 4;
                r = 255;
                g = Math.floor(255 * (1 - ratio) + 100 * ratio);
                b = 0;
            }}

            return `rgb(${{r}},${{g}},${{b}})`;
        }}

        function updateDisplay() {{
            const snapshot = snapshots[currentIndex];

            // Update info panel
            document.getElementById('currentMove').textContent = snapshot.move;
            document.getElementById('shipsRemaining').textContent = snapshot.ships_remaining;

            let resultText = snapshot.result;
            if (snapshot.ship_sunk) {{
                resultText += ` - ${{snapshot.ship_sunk.toUpperCase()}} SUNK!`;
            }}
            document.getElementById('lastResult').textContent = resultText;

            // Update attack board
            for (let i = 0; i < rows; i++) {{
                for (let j = 0; j < cols; j++) {{
                    const cell = document.getElementById(`attack_${{i}}_${{j}}`);
                    const value = snapshot.attack_board[i][j];
                    cell.style.backgroundColor = getAttackCellColor(value);

                    // Highlight action
                    const actionRow = Math.floor(snapshot.action / cols);
                    const actionCol = snapshot.action % cols;
                    if (i === actionRow && j === actionCol) {{
                        cell.style.border = '3px solid #f39c12';
                    }} else {{
                        cell.style.border = '1px solid #2c3e50';
                    }}
                }}
            }}

            // Update probability board
            if (snapshot.normalized_prob) {{
                for (let i = 0; i < rows; i++) {{
                    for (let j = 0; j < cols; j++) {{
                        const cell = document.getElementById(`prob_${{i}}_${{j}}`);
                        const value = snapshot.normalized_prob[i][j];
                        const rawValue = snapshot.probability_grid[i][j];

                        cell.style.backgroundColor = getHeatmapColor(value);
                        cell.title = `Probability: ${{rawValue.toFixed(1)}}`;

                        // Show max value
                        if (value === 1.0 && rawValue > 0) {{
                            cell.innerHTML = `<strong>${{Math.round(rawValue)}}</strong>`;
                            cell.style.color = 'white';
                        }} else {{
                            cell.innerHTML = '';
                        }}

                        // Highlight action
                        const actionRow = Math.floor(snapshot.action / cols);
                        const actionCol = snapshot.action % cols;
                        if (i === actionRow && j === actionCol) {{
                            cell.style.border = '3px solid #f39c12';
                        }} else {{
                            cell.style.border = '1px solid #2c3e50';
                        }}
                    }}
                }}
            }}

            // Update slider
            document.getElementById('moveSlider').value = currentIndex;

            // Update button states
            document.getElementById('firstBtn').disabled = currentIndex === 0;
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === snapshots.length - 1;
            document.getElementById('lastBtn').disabled = currentIndex === snapshots.length - 1;
        }}

        function nextMove() {{
            if (currentIndex < snapshots.length - 1) {{
                currentIndex++;
                updateDisplay();
            }} else if (isPlaying) {{
                stopPlay();
            }}
        }}

        function previousMove() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateDisplay();
            }}
        }}

        function goToFirst() {{
            currentIndex = 0;
            updateDisplay();
        }}

        function goToLast() {{
            currentIndex = snapshots.length - 1;
            updateDisplay();
        }}

        function sliderChanged(value) {{
            currentIndex = parseInt(value);
            updateDisplay();
        }}

        function togglePlay() {{
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }}

        function startPlay() {{
            isPlaying = true;
            document.getElementById('playBtn').textContent = '⏸ Pause';
            const speed = parseInt(document.getElementById('speedSlider').value);
            playInterval = setInterval(nextMove, speed);
        }}

        function stopPlay() {{
            isPlaying = false;
            document.getElementById('playBtn').textContent = '▶ Play';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}

        // Speed slider handler
        document.getElementById('speedSlider').addEventListener('input', function(e) {{
            const speed = parseInt(e.target.value);
            document.getElementById('speedLabel').textContent = (speed / 1000).toFixed(1) + 's';

            if (isPlaying) {{
                stopPlay();
                startPlay();
            }}
        }});

        // Initialize
        initBoards();
        updateDisplay();
    </script>
</body>
</html>
"""


def record_game_snapshots(
    env,
    agent: ProbabilityAgent,
    seed: int = 42
) -> Tuple[List[GameSnapshot], dict]:
    """
    Record all snapshots from a complete game.

    Args:
        env: BattleshipEnv instance
        agent: ProbabilityAgent instance
        seed: Random seed

    Returns:
        Tuple of (snapshots list, game stats dict)
    """
    snapshots = []

    obs, info = env.reset(seed=seed)
    agent.reset()

    total_ships = info['total_ships']
    done = False
    step_count = 0

    # Initial snapshot (before any moves)
    agent.select_action(obs)  # Calculate probabilities
    snapshots.append(GameSnapshot(
        move_number=0,
        attack_board=obs['attack_board'],
        probability_grid=agent.get_probability_grid(),
        action=-1,
        result="start",
        ships_remaining=total_ships
    ))

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1

        snapshots.append(GameSnapshot(
            move_number=step_count,
            attack_board=obs['attack_board'],
            probability_grid=agent.get_probability_grid(),
            action=action,
            result=info.get('result', 'unknown'),
            ship_sunk=info.get('ship_sunk'),
            ships_remaining=info.get('ships_remaining', 0)
        ))

        done = terminated or truncated

    stats = {
        'moves': step_count,
        'victory': terminated,
        'total_ships': total_ships
    }

    return snapshots, stats
