"""
LLM-based Battleship agent using HuggingFace transformers.

Supports both inference and fine-tuning workflows.
"""

from typing import Dict, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from game.agents.base import BattleshipAgent


class LLMAgent(BattleshipAgent):
    """
    LLM agent that converts board state to text prompt and parses response.

    Supports:
    - Zero-shot inference with base models
    - Fine-tuned models (SFT/GRPO)
    - Temperature/sampling controls
    - Fallback to random valid action for invalid outputs
    """

    def __init__(
        self,
        name: str,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "mps",  # or "cuda", "cpu"
        max_new_tokens: int = 20,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        use_flash_attention: bool = False,
    ):
        super().__init__(name)
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load model and tokenizer
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        }

        # Only use device_map for GPU devices
        if device in ["cuda", "mps"]:
            model_kwargs["device_map"] = device

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Move to device if CPU
        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        print(f"✓ Model loaded successfully")

        # State tracking
        self.board_size = None
        self.conversation_history = []

    def select_action(self, observation: Dict[str, np.ndarray], env=None) -> int:
        """
        Convert observation to text, query LLM, parse response to action.

        Falls back to random valid action if LLM output is invalid.
        """
        attack_board = observation['attack_board']
        remaining_ships = observation['remaining_ships']
        move_count = observation['move_count'][0]

        rows, cols = attack_board.shape
        if self.board_size is None:
            self.board_size = (rows, cols)

        # 1. Convert observation to text
        prompt = self._build_prompt(attack_board, remaining_ships, move_count)

        # 2. Generate LLM response
        response = self._generate(prompt)

        # 3. Parse response to action
        action = self._parse_response(response, attack_board)

        # 4. Validate and fallback if needed
        if not self._is_valid_action(action, attack_board):
            action = self._get_random_valid_action(attack_board)
            if env and env.verbose:
                print(f"⚠️  Invalid LLM output '{response}', using fallback action")

        return action

    def _build_prompt(self, attack_board: np.ndarray, remaining_ships: np.ndarray,
                     move_count: int) -> str:
        """Convert game state to LLM prompt."""
        rows, cols = attack_board.shape

        # Build board visualization
        board_lines = []
        board_lines.append("   " + " ".join(str(c) for c in range(cols)))
        for r in range(rows):
            row_str = f"{r}  "
            for c in range(cols):
                cell = attack_board[r, c]
                if cell == 0:
                    row_str += ". "  # Unknown
                elif cell == 1:
                    row_str += "O "  # Miss
                else:  # cell == 2
                    row_str += "X "  # Hit
            board_lines.append(row_str)

        board_str = "\n".join(board_lines)

        # Get unsunk ship sizes
        ships_remaining = [int(s) for s in remaining_ships if s > 0]

        # Detect if in target mode (have hits but not all adjacent cells are attacked)
        hit_positions = list(zip(*np.where(attack_board == 2)))
        in_target_mode = len(hit_positions) > 0

        # Build system message
        system = """You are an expert Battleship player. Your goal is to sink all enemy ships efficiently.

Strategy:
- HUNT MODE: When no hits, use checkerboard pattern (skip adjacent cells) for efficient coverage
- TARGET MODE: When you hit a ship, attack adjacent cells (up/down/left/right) to find the rest
- Think about ship orientations (horizontal or vertical)
- Remember ship sizes and eliminate impossible placements

Output format: Respond with ONLY the coordinates as "row,col" (e.g., "3,4"). No explanations."""

        # Build user message
        user_msg = f"""Move #{move_count + 1}

Board ({rows}x{cols}):
{board_str}

Legend: . = unknown, O = miss, X = hit
Remaining ship sizes: {ships_remaining}

{"[TARGET MODE] You have hits! Attack adjacent cells to find the rest of the ship." if in_target_mode else "[HUNT MODE] Search efficiently using checkerboard pattern."}

What cell should I attack next? (format: row,col)"""

        # Format for instruction-following models
        if "Instruct" in self.model_name or "instruct" in self.model_name:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Base models without chat template
            prompt = f"{system}\n\n{user_msg}\n\nAnswer:"

        return prompt

    def _generate(self, prompt: str) -> str:
        """Generate LLM response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (not the prompt)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    def _parse_response(self, response: str, attack_board: np.ndarray) -> Optional[int]:
        """
        Parse LLM response to action index.

        Handles formats:
        - "3,4" -> row 3, col 4
        - "row 3, col 4"
        - "(3, 4)"
        """
        rows, cols = attack_board.shape

        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+', response)

        if len(numbers) >= 2:
            row = int(numbers[0])
            col = int(numbers[1])

            # Validate bounds
            if 0 <= row < rows and 0 <= col < cols:
                return row * cols + col

        return None

    def _is_valid_action(self, action: Optional[int], attack_board: np.ndarray) -> bool:
        """Check if action is valid (in bounds and unattacked)."""
        if action is None:
            return False

        rows, cols = attack_board.shape
        if not (0 <= action < rows * cols):
            return False

        row, col = divmod(action, cols)
        return attack_board[row, col] == 0  # Unattacked

    def _get_random_valid_action(self, attack_board: np.ndarray) -> int:
        """Fallback: pick random valid action."""
        rows, cols = attack_board.shape
        valid_actions = [
            r * cols + c
            for r in range(rows)
            for c in range(cols)
            if attack_board[r, c] == 0
        ]
        return int(np.random.choice(valid_actions))

    def reset(self):
        """Reset between episodes."""
        self.conversation_history = []
        self.board_size = None
