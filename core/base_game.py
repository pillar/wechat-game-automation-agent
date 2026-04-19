from abc import ABC, abstractmethod
from PIL.Image import Image
from typing import Dict, Any


class BaseGame(ABC):
    """Abstract base class for game adapters.

    Subclasses must implement the three core methods:
    - build_prompt: Generate prompt for AI analysis
    - parse_ai_response: Parse AI response into action dict
    - execute_action: Perform actual game interaction
    """

    # ── Core methods (must implement) ──────────────────────────────

    @abstractmethod
    def build_prompt(self, screenshot: Image) -> str:
        """Generate prompt for AI analysis based on current screenshot.

        Args:
            screenshot: PIL Image of the current game screen

        Returns:
            String prompt to send to Gemini Vision API
        """
        pass

    @abstractmethod
    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response and return structured action.

        Args:
            response: Raw text response from Gemini Vision API

        Returns:
            Action dict with keys like 'action', 'x', 'y', 'duration', etc.
        """
        pass

    @abstractmethod
    def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute the action on the game.

        Args:
            action: Action dict returned by parse_ai_response
        """
        pass

    # ── Hook methods (optional override) ──────────────────────────

    def on_round_start(self) -> None:
        """Called at the beginning of each round.

        Override to add pre-action delays, wait for animations, etc.
        """
        pass

    def on_round_end(self, action: Dict[str, Any], success: bool) -> None:
        """Called at the end of each round.

        Args:
            action: The action that was executed
            success: Whether the action was executed without errors
        """
        pass

    def is_game_over(self, screenshot: Image) -> bool:
        """Check if the game has ended.

        Args:
            screenshot: Current game screenshot

        Returns:
            True if game is over, False otherwise
        """
        return False

    def on_game_over(self) -> bool:
        """Called when game is over. Handle restart if needed.

        Returns:
            True if restarted and should continue, False if should stop
        """
        return False

    def get_game_name(self) -> str:
        """Get human-readable game name.

        Returns:
            Game name string
        """
        return self.__class__.__name__
