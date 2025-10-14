"""Tests for ship placement logic."""

import pytest
import numpy as np
from game.placement import (
    place_ships,
    validate_manual_placement,
    _is_valid_placement,
    _get_buffer_zone,
    _is_straight_line
)
from game.config import get_ship_config


class TestShipPlacement:
    """Test ship placement with no-touch constraint."""

    def test_ships_placed_successfully(self):
        """Test that all ships are placed on a 10x10 board."""
        board_size = (10, 10)
        ship_config = get_ship_config(board_size)
        rng = np.random.default_rng(42)

        ships, ship_board = place_ships(board_size, ship_config, rng)

        assert len(ships) == len(ship_config)
        for ship_name, ship_size in ship_config.items():
            assert ship_name in ships
            assert ships[ship_name].size == ship_size

    def test_ships_dont_overlap(self):
        """Test that no two ships overlap."""
        board_size = (10, 10)
        ship_config = get_ship_config(board_size)
        rng = np.random.default_rng(42)

        ships, ship_board = place_ships(board_size, ship_config, rng)

        # Check each cell has at most one ship
        all_coords = []
        for ship in ships.values():
            all_coords.extend(ship.coords)

        assert len(all_coords) == len(set(all_coords)), "Ships overlap!"

    def test_ships_dont_touch(self):
        """Test that no two ships touch (including diagonally)."""
        board_size = (10, 10)
        ship_config = get_ship_config(board_size)
        rng = np.random.default_rng(42)

        ships, ship_board = place_ships(board_size, ship_config, rng)

        ship_list = list(ships.values())
        for i, ship1 in enumerate(ship_list):
            buffer1 = _get_buffer_zone(ship1.coords, board_size)
            for ship2 in ship_list[i+1:]:
                # No coord from ship2 should be in buffer of ship1 (except ship1 itself)
                for coord in ship2.coords:
                    assert coord not in buffer1, (
                        f"Ship {ship1.name} and {ship2.name} are touching! "
                        f"Coord {coord} from {ship2.name} is in buffer of {ship1.name}"
                    )

    def test_ships_within_bounds(self):
        """Test that all ships are within board bounds."""
        board_size = (10, 10)
        ship_config = get_ship_config(board_size)
        rng = np.random.default_rng(42)

        ships, ship_board = place_ships(board_size, ship_config, rng)

        rows, cols = board_size
        for ship in ships.values():
            for r, c in ship.coords:
                assert 0 <= r < rows, f"Row {r} out of bounds"
                assert 0 <= c < cols, f"Col {c} out of bounds"

    def test_ships_form_straight_lines(self):
        """Test that all ships form straight horizontal or vertical lines."""
        board_size = (10, 10)
        ship_config = get_ship_config(board_size)
        rng = np.random.default_rng(42)

        ships, ship_board = place_ships(board_size, ship_config, rng)

        for ship in ships.values():
            assert _is_straight_line(ship.coords), (
                f"Ship {ship.name} does not form straight line: {ship.coords}"
            )

    def test_small_board_placement(self):
        """Test placement on a 5x5 board (minimum size)."""
        board_size = (5, 5)
        ship_config = get_ship_config(board_size)
        rng = np.random.default_rng(42)

        ships, ship_board = place_ships(board_size, ship_config, rng)

        assert len(ships) == 2  # Should have destroyer and cruiser
        assert "destroyer" in ships
        assert "cruiser" in ships

    def test_ship_board_values(self):
        """Test that ship_board has correct values."""
        board_size = (10, 10)
        ship_config = get_ship_config(board_size)
        rng = np.random.default_rng(42)

        ships, ship_board = place_ships(board_size, ship_config, rng)

        # Count non-zero cells
        total_ship_cells = sum(ship.size for ship in ships.values())
        assert np.count_nonzero(ship_board) == total_ship_cells

        # Check each ship ID appears correct number of times
        for ship_id in range(1, len(ships) + 1):
            assert np.count_nonzero(ship_board == ship_id) > 0


class TestManualPlacement:
    """Test manual placement validation."""

    def test_valid_manual_placement(self):
        """Test that valid manual placement passes validation."""
        board_size = (10, 10)
        ship_coords = {
            "destroyer": [(0, 0), (0, 1)],
            "cruiser": [(3, 3), (3, 4), (3, 5)]
        }

        # Should not raise
        validate_manual_placement(ship_coords, board_size)

    def test_overlapping_ships_rejected(self):
        """Test that overlapping ships are rejected."""
        board_size = (10, 10)
        ship_coords = {
            "destroyer": [(0, 0), (0, 1)],
            "cruiser": [(0, 1), (0, 2), (0, 3)]  # Overlaps at (0, 1)
        }

        with pytest.raises(ValueError, match="overlaps"):
            validate_manual_placement(ship_coords, board_size)

    def test_touching_ships_rejected(self):
        """Test that touching ships are rejected."""
        board_size = (10, 10)
        ship_coords = {
            "destroyer": [(0, 0), (0, 1)],
            "cruiser": [(0, 2), (0, 3), (0, 4)]  # Touches destroyer
        }

        with pytest.raises(ValueError, match="touching"):
            validate_manual_placement(ship_coords, board_size)

    def test_out_of_bounds_rejected(self):
        """Test that out-of-bounds coordinates are rejected."""
        board_size = (10, 10)
        ship_coords = {
            "destroyer": [(0, 0), (0, 1)],
            "cruiser": [(10, 0), (10, 1), (10, 2)]  # Row 10 is out of bounds
        }

        with pytest.raises(ValueError, match="out of bounds"):
            validate_manual_placement(ship_coords, board_size)

    def test_non_straight_line_rejected(self):
        """Test that non-straight ships are rejected."""
        board_size = (10, 10)
        ship_coords = {
            "destroyer": [(0, 0), (0, 1)],
            "cruiser": [(2, 2), (2, 3), (3, 3)]  # L-shape
        }

        with pytest.raises(ValueError, match="straight line"):
            validate_manual_placement(ship_coords, board_size)


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_straight_line_horizontal(self):
        """Test horizontal line detection."""
        coords = [(0, 0), (0, 1), (0, 2)]
        assert _is_straight_line(coords)

    def test_is_straight_line_vertical(self):
        """Test vertical line detection."""
        coords = [(0, 0), (1, 0), (2, 0)]
        assert _is_straight_line(coords)

    def test_is_straight_line_single_cell(self):
        """Test single cell is considered straight."""
        coords = [(0, 0)]
        assert _is_straight_line(coords)

    def test_is_straight_line_diagonal_rejected(self):
        """Test diagonal is not straight."""
        coords = [(0, 0), (1, 1), (2, 2)]
        assert not _is_straight_line(coords)

    def test_is_straight_line_l_shape_rejected(self):
        """Test L-shape is not straight."""
        coords = [(0, 0), (0, 1), (1, 1)]
        assert not _is_straight_line(coords)

    def test_buffer_zone_includes_neighbors(self):
        """Test that buffer zone includes all 8 neighbors."""
        board_size = (10, 10)
        coords = [(5, 5)]  # Single cell in middle

        buffer = _get_buffer_zone(coords, board_size)

        # Should have 9 cells: original + 8 neighbors
        assert len(buffer) == 9
        assert (5, 5) in buffer
        # Check all 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                assert (5 + dr, 5 + dc) in buffer

    def test_buffer_zone_respects_bounds(self):
        """Test that buffer zone doesn't go out of bounds."""
        board_size = (5, 5)
        coords = [(0, 0)]  # Corner cell

        buffer = _get_buffer_zone(coords, board_size)

        # Should only have 4 cells: (0,0), (0,1), (1,0), (1,1)
        assert len(buffer) == 4
        for r, c in buffer:
            assert 0 <= r < 5
            assert 0 <= c < 5
