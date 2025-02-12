import pygame
import numpy as np
from game.core import _HIT_IDX, _MISS_IDX, _SHIP_IDX, _EMPTY_IDX

class BattleshipVisualizer:
    def __init__(self, cell_size=40):
        pygame.init()
        self.cell_size = cell_size
        self.colors = {
            _EMPTY_IDX: (200, 200, 255),  # Light blue for water
            _SHIP_IDX: (128, 128, 128),   # Gray for ships
            _HIT_IDX: (255, 0, 0),        # Red for hits
            _MISS_IDX: (255, 255, 255)    # White for misses
        }
        self.font = pygame.font.Font(None, 36)
        
    def initialize_display(self, board_sizes):
        """Initialize the display based on board sizes."""
        width = (board_sizes[0][1] + board_sizes[1][1]) * self.cell_size + 3 * self.cell_size
        height = max(board_sizes[0][0], board_sizes[1][0]) * self.cell_size + 2 * self.cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Battleship Game")

    def draw_grid(self, surface, start_x, start_y, rows, cols):
        """Draw the grid lines."""
        for i in range(rows + 1):
            pygame.draw.line(surface, (0, 0, 0), 
                           (start_x, start_y + i * self.cell_size),
                           (start_x + cols * self.cell_size, start_y + i * self.cell_size))
        for j in range(cols + 1):
            pygame.draw.line(surface, (0, 0, 0),
                           (start_x + j * self.cell_size, start_y),
                           (start_x + j * self.cell_size, start_y + rows * self.cell_size))

    def draw_board(self, board, hit_board, start_x, start_y):
        """Draw a single player's board."""
        rows, cols = board.shape
        for i in range(rows):
            for j in range(cols):
                rect = pygame.Rect(start_x + j * self.cell_size,
                                 start_y + i * self.cell_size,
                                 self.cell_size, self.cell_size)
                
                # Draw cell color based on state
                if hit_board[i, j] == _HIT_IDX:
                    color = self.colors[_HIT_IDX]
                elif hit_board[i, j] == _MISS_IDX:
                    color = self.colors[_MISS_IDX]
                elif board[i, j] == _SHIP_IDX:
                    color = self.colors[_SHIP_IDX]
                else:
                    color = self.colors[_EMPTY_IDX]
                    
                pygame.draw.rect(self.screen, color, rect)

    def visualize_state(self, state):
        """Visualize the current game state."""
        self.screen.fill((230, 230, 230))  # Light gray background

        # Draw Player 1's board
        start_x1 = self.cell_size
        start_y1 = self.cell_size
        self.draw_board(state.boards[0], state.hit_boards[0], start_x1, start_y1)
        self.draw_grid(self.screen, start_x1, start_y1, state.sizes[0][0], state.sizes[0][1])

        # Draw Player 2's board
        start_x2 = start_x1 + state.sizes[0][1] * self.cell_size + self.cell_size
        start_y2 = self.cell_size
        self.draw_board(state.boards[1], state.hit_boards[1], start_x2, start_y2)
        self.draw_grid(self.screen, start_x2, start_y2, state.sizes[1][0], state.sizes[1][1])

        # Add labels
        label1 = self.font.render("Player 1", True, (0, 0, 0))
        label2 = self.font.render("Player 2", True, (0, 0, 0))
        self.screen.blit(label1, (start_x1, 10))
        self.screen.blit(label2, (start_x2, 10))

        pygame.display.flip()

    def close(self):
        """Close the PyGame window."""
        pygame.quit() 