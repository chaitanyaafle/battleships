"""Ship placement logic with no-touch constraint."""

import numpy as np
from typing import Dict, List, Tuple, Set
from game.state import Ship


def place_ships(
    board_size: Tuple[int, int],
    ship_config: Dict[str, int],
    rng: np.random.Generator,
    allow_adjacent: bool = False
) -> Tuple[Dict[str, Ship], np.ndarray]:
    """
    Randomly place ships on board.

    Args:
        board_size: (rows, cols) tuple
        ship_config: Dictionary mapping ship names to sizes
        rng: Random number generator (for seeding)
        allow_adjacent: If True, ships can touch (including diagonally).
                       If False (default), enforces no-touch constraint.

    Returns:
        Tuple of:
            - ships: Dictionary of Ship objects
            - ship_board: Board array with ship positions (0=water, ship_id>0)

    Raises:
        RuntimeError: If unable to place ships after max attempts
    """
    rows, cols = board_size
    ship_board = np.zeros((rows, cols), dtype=np.int32)
    occupied_cells: Set[Tuple[int, int]] = set()
    ships = {}

    # Sort ships by size (largest first for easier placement)
    sorted_ships = sorted(ship_config.items(), key=lambda x: x[1], reverse=True)

    for ship_id, (ship_name, size) in enumerate(sorted_ships, start=1):
        max_attempts = 1000
        placed = False

        for _ in range(max_attempts):
            # Random orientation and position
            horizontal = rng.choice([True, False])

            if horizontal:
                row = rng.integers(0, rows)
                col = rng.integers(0, cols - size + 1)
                coords = [(row, col + i) for i in range(size)]
            else:
                row = rng.integers(0, rows - size + 1)
                col = rng.integers(0, cols)
                coords = [(row + i, col) for i in range(size)]

            # Check if placement is valid (no overlap or touching)
            if _is_valid_placement(coords, occupied_cells, board_size, allow_adjacent):
                # Place ship
                for r, c in coords:
                    ship_board[r, c] = ship_id

                # Mark occupied cells (including buffer zone if needed)
                if allow_adjacent:
                    # Only mark the ship's actual cells
                    occupied_cells.update(coords)
                else:
                    # Mark ship cells + buffer zone for no-touch constraint
                    occupied_cells.update(_get_buffer_zone(coords, board_size))

                ships[ship_name] = Ship(
                    name=ship_name,
                    size=size,
                    coords=coords
                )

                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Failed to place {ship_name} (size {size}) after {max_attempts} attempts. "
                f"Board size {rows}x{cols} may be too small for ship configuration. "
                f"Already placed: {list(ships.keys())}"
            )

    return ships, ship_board


def _is_valid_placement(
    coords: List[Tuple[int, int]],
    occupied: Set[Tuple[int, int]],
    board_size: Tuple[int, int],
    allow_adjacent: bool = False
) -> bool:
    """
    Check if ship placement is valid (within bounds, no overlap).

    Args:
        coords: List of (row, col) coordinates for ship
        occupied: Set of already occupied cells (includes buffer zones if no-touch)
        board_size: (rows, cols) tuple
        allow_adjacent: If True, only check for overlaps (not touching)

    Returns:
        True if placement is valid, False otherwise
    """
    rows, cols = board_size

    if allow_adjacent:
        # Only check for direct overlaps (ships can touch)
        for r, c in coords:
            # Check if in bounds
            if not (0 <= r < rows and 0 <= c < cols):
                return False
            # Check if cell is already occupied by another ship
            if (r, c) in occupied:
                return False
    else:
        # Check for overlaps and touching (no-touch constraint)
        for r, c in coords:
            # Check if in bounds
            if not (0 <= r < rows and 0 <= c < cols):
                return False

            # Check 8 neighbors + self for any occupation
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) in occupied:
                            return False

    return True


def _get_buffer_zone(
    coords: List[Tuple[int, int]],
    board_size: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """
    Get all cells occupied by ship plus surrounding buffer (8-neighbors).

    Args:
        coords: List of (row, col) coordinates for ship
        board_size: (rows, cols) tuple

    Returns:
        Set of all cells in ship and its buffer zone
    """
    rows, cols = board_size
    buffer = set(coords)

    for r, c in coords:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    buffer.add((nr, nc))

    return buffer


def validate_manual_placement(
    ship_coords: Dict[str, List[Tuple[int, int]]],
    board_size: Tuple[int, int]
) -> None:
    """
    Validate manually specified ship placement.

    Args:
        ship_coords: Dictionary mapping ship names to coordinate lists
        board_size: (rows, cols) tuple

    Raises:
        ValueError: If placement is invalid
    """
    rows, cols = board_size
    all_coords: Set[Tuple[int, int]] = set()

    for ship_name, coords in ship_coords.items():
        # Check ship has at least 1 cell
        if not coords:
            raise ValueError(f"Ship {ship_name} has no coordinates")

        # Check all coordinates are in bounds
        for r, c in coords:
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(
                    f"Ship {ship_name} coordinate ({r}, {c}) out of bounds "
                    f"for board size {rows}x{cols}"
                )

        # Check ship forms a straight line
        if not _is_straight_line(coords):
            raise ValueError(
                f"Ship {ship_name} coordinates do not form a straight line: {coords}"
            )

        # Check for overlaps
        for coord in coords:
            if coord in all_coords:
                raise ValueError(
                    f"Ship {ship_name} overlaps with another ship at {coord}"
                )
            all_coords.add(coord)

    # Check no-touch constraint
    ship_list = list(ship_coords.values())
    for i, coords1 in enumerate(ship_list):
        buffer1 = _get_buffer_zone(coords1, board_size)
        for coords2 in ship_list[i+1:]:
            # Check if any coord from ship2 is in buffer of ship1
            if any(coord in buffer1 for coord in coords2):
                raise ValueError(
                    "Ships are touching (no-touch constraint violated). "
                    "Ships must have at least 1 cell gap between them (including diagonals)."
                )


def _is_straight_line(coords: List[Tuple[int, int]]) -> bool:
    """
    Check if coordinates form a straight horizontal or vertical line.

    Args:
        coords: List of (row, col) coordinates

    Returns:
        True if coordinates form a straight line, False otherwise
    """
    if len(coords) <= 1:
        return True

    # Sort coordinates
    sorted_coords = sorted(coords)

    # Check if all rows are same (horizontal)
    rows = [r for r, c in sorted_coords]
    cols = [c for r, c in sorted_coords]

    if len(set(rows)) == 1:
        # Horizontal: check columns are consecutive
        return cols == list(range(cols[0], cols[0] + len(cols)))
    elif len(set(cols)) == 1:
        # Vertical: check rows are consecutive
        return rows == list(range(rows[0], rows[0] + len(rows)))
    else:
        return False
