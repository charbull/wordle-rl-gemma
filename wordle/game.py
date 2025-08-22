from dataclasses import dataclass, field
from typing import List

# ==============================================================================
# ---  DATA STRUCTURES FOR GAME LOGIC ---
# ==============================================================================
@dataclass
class GameRecord:
    """A structured object to hold the results of a single game."""
    log_type: str
    step: int
    secret_word: str
    solved: bool
    turns_to_solve: int
    final_reward: float
    loss_at_step: float

@dataclass
class GuessFeedback:
    """Represents a single guess and its corresponding feedback."""
    guess: str
    feedback: str
    is_in_dictionary: bool = True

@dataclass
class GenerationAttempt:
    """Stores all information related to a single model generation attempt in a game."""
    prompt_string: str
    prompt_tokens: list
    full_response: str
    response_tokens: list
    parsed_guess: str
    feedback_given: GuessFeedback
    game_score: float # The score based only on the guess quality
    training_reward: float # The final reward for the RL trainer


@dataclass
class GameRollout:
    """Contains the full history and outcome of one played game."""
    attempts: List[GenerationAttempt] = field(default_factory=list)
    secret_word: str = ""
    solved: bool = False