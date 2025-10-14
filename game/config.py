"""Ship configuration system based on board size."""

from typing import Dict, Tuple


def get_ship_config(board_size: Tuple[int, int]) -> Dict[str, int]:
    """
    Get ship configuration based on board size.

    Ship configurations are adaptive:
    - 5x5 to 7x7: Destroyer(2), Cruiser(3)
    - 8x8 to 9x9: + Submarine(3)
    - 10x10+:     + Battleship(4), Carrier(5)

    Args:
        board_size: (rows, cols) tuple

    Returns:
        Dictionary mapping ship names to sizes

    Raises:
        ValueError: If board size is too small (< 5x5)
    """
    rows, cols = board_size
    min_dim = min(rows, cols)

    if min_dim < 5:
        raise ValueError(
            f"Board size must be at least 5x5. Got {rows}x{cols} "
            f"(min dimension: {min_dim})"
        )

    if min_dim <= 7:
        return {
            "destroyer": 2,
            "cruiser": 3
        }
    elif min_dim <= 9:
        return {
            "destroyer": 2,
            "cruiser": 3,
            "submarine": 3
        }
    else:  # 10+
        return {
            "carrier": 5,
            "battleship": 4,
            "cruiser": 3,
            "submarine": 3,
            "destroyer": 2
        }


def validate_board_size(board_size: Tuple[int, int]) -> None:
    """
    Validate that board size is reasonable.

    Args:
        board_size: (rows, cols) tuple

    Raises:
        ValueError: If board size is invalid
    """
    rows, cols = board_size

    if rows < 5 or cols < 5:
        raise ValueError(f"Board dimensions must be at least 5. Got {rows}x{cols}")

    if rows > 20 or cols > 20:
        raise ValueError(
            f"Board dimensions must be at most 20. Got {rows}x{cols}. "
            "Larger boards may cause performance issues."
        )

    # Check if board is large enough for ships
    ship_config = get_ship_config(board_size)
    total_ship_cells = sum(ship_config.values())
    board_area = rows * cols

    # With no-touch constraint, we need roughly 3x the ship cells
    # (each ship cell needs ~1 buffer cell around it)
    min_required_area = total_ship_cells * 3

    if board_area < min_required_area:
        raise ValueError(
            f"Board size {rows}x{cols} ({board_area} cells) is too small "
            f"for {len(ship_config)} ships ({total_ship_cells} cells). "
            f"Minimum recommended area: {min_required_area} cells. "
            "Ships may fail to place with no-touch constraint."
        )
