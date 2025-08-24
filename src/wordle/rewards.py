from collections import Counter
from typing import List
from src.utils import config as cfg
from collections import Counter
from src.wordle.game import GuessFeedback, parse_guess, is_valid_guess, find_valid_completions, get_clue_summary, get_feedback
from src.utils import constants




def reward_for_mid_game_exploration(guess: str, known_letters: set, config: cfg.TrainerConfig) -> float:
    """
    Provides a reward bonus for using new, unknown letters on turns 2-6.
    This acts as a proxy for information gain when the solution space is smaller.
    
    Args:
        guess (str): The current guess.
        known_letters (set): A set of all letters that have appeared in previous
                             guesses (greens, yellows, and grays).
        config (cfg.TrainerConfig): The trainer configuration.
        
    Returns:
        float: The calculated exploration bonus.
    """
    if not guess:
        return 0.0
    
    # Find the letters in the current guess that have not been seen before.
    new_letters = set(guess.upper()) - known_letters
    num_new_letters = len(new_letters)
    
    # Provide a bonus for each new letter introduced.
    return num_new_letters * config.reward.get("new_letter_bonus", 0.0)


def get_strategic_bonus(guess: str, past_feedback: List[GuessFeedback], config: cfg.TrainerConfig) -> float:
    """
    Determines the appropriate strategic bonus based on the turn number.
    - Turn 1: Uses pre-calculated, perfect information gain.
    - Turns 2-6: Uses a proxy that rewards exploring new letters.
    """
    turn_number = len(past_feedback) + 1
    
    if turn_number == 1:
        # On the first turn, use the most powerful tool we have.
        return reward_for_information_gain(guess, config)
    else:
        # On subsequent turns, reward exploration of new letters.
        
        # First, gather all letters seen in previous guesses.
        known_letters = set()
        for fb in past_feedback:
            known_letters.update(fb.guess.upper())
            
        return reward_for_mid_game_exploration(guess, known_letters, config)

def reward_for_information_gain(guess: str, config: cfg.TrainerConfig) -> float:
    """Provides a reward based on pre-calculated information gain (entropy)."""
    if not guess:
        return 0.0
    entropy_score = constants.WORD_ENTROPY_SCORES.get(guess.upper(), 0.0)
    return entropy_score * config.reward.get("information_gain_bonus_coeff")


def calculate_stagnation_penalty(guess: str, known_green: dict, known_yellow: Counter, config: cfg.TrainerConfig) -> float:
    """
    Calculates a penalty for re-using known information inefficiently.
    - Penalizes placing a known green letter in its correct spot again.
    - Penalizes re-using any known yellow letter.
    """
    penalty = 0.0
    
    # Penalize reusing green letters in their known correct positions
    green_reuse_penalty = config.reward.get("green_reuse_penalty")
    for idx, letter in known_green.items():
        if guess[idx] == letter:
            penalty += green_reuse_penalty

    # Penalize reusing yellow letters (encourages using new letters to find other yellows/greens)
    yellow_reuse_penalty = config.reward.get("yellow_reuse_penalty")
    for letter in set(guess):
        if letter in known_yellow:
            penalty += yellow_reuse_penalty
            
    return penalty

def reward_for_possibility_reduction(
    past_feedback: List[GuessFeedback], 
    new_feedback: GuessFeedback, 
    config: cfg.TrainerConfig
) -> float:
    """
    Calculates a reward based on how much a guess reduces the number of
    possible remaining answers.
    """
    if not past_feedback: # Not applicable for the first real guess
        return 0.0

    # Find possibilities BEFORE the new guess
    clues_before = get_clue_summary([f.guess for f in past_feedback], [f.feedback for f in past_feedback])
    possibilities_before = find_valid_completions(clues_before, constants.ANSWERS_WORDS)
    
    # Find possibilities AFTER the new guess
    combined_feedback = past_feedback + [new_feedback]
    clues_after = get_clue_summary([f.guess for f in combined_feedback], [f.feedback for f in combined_feedback])
    possibilities_after = find_valid_completions(clues_after, constants.ANSWERS_WORDS)

    # The bonus is proportional to the percentage of possibilities eliminated
    if not possibilities_before: return 0.0
    
    reduction_fraction = (len(possibilities_before) - len(possibilities_after)) / len(possibilities_before)
    
    return reduction_fraction * config.reward.get("possibility_reduction_bonus", 10.0)


def calculate_total_reward(
    response: str,
    secret_word: str,
    past_feedback: List[GuessFeedback],
    config: cfg.TrainerConfig,
    allowed_words: set,
    tokenizer = None,
    print_debug: bool = False
) -> tuple[float, float]: # Return a tuple of (game_score, training_reward)
    """
    Calculates a training_reward and a game score for a given model response.
    - Prioritizes winning.
    - Penalizes invalid/repeated guesses.
    - Applies penalties for violating known clues.
    - Rewards valid, consistent guesses with a base reward and an entropy.
    - Applies a small time penalty to every guess to encourage efficiency.

    Args:
        response: The full text response from the model (after the prompt).
        secret_word: The secret word to be guessed in this game.
        past_feedback: List of GuessFeedback objects representing previous guesses and feedback.
        config: The TrainerConfig containing reward parameters.
        allowed_words: Set of valid 5-letter words for guesses.
        tokenizer: The tokenizer to calculate length penalties.
        print_debug: If True, prints detailed debug information about the reward calculation.
    Returns:
        A tuple of (game_score, training_reward). The game_score is based solely on the guess quality, 
            the training_reward includes time and length penalties for the loss function.
    """
    reward_config = config.reward
    guess = parse_guess(response)
    # we want the model to win in the fewest guesses possible
    time_penalty = -reward_config.get("time_penalty_per_guess")
    # we want the model to be concise in the reasoning
    length_penalty = 0.0
    if tokenizer and response:
        length_penalty = -(len(tokenizer.encode(response)) * reward_config.get("length_penalty_per_token"))

    # Handle Game Rule Violations
    if not guess:
        # This ensures that a short, empty response is not rewarded over a
        # long but incorrect attempt. 
        # avoid reward hacking by returning a very negative score
        game_score = -reward_config.get("format_fail_penalty")
        training_reward = game_score + time_penalty + length_penalty    
        return game_score, training_reward
    
    if guess == secret_word.upper():
        game_score = reward_config.get("solution_correct_guess")
        return game_score, game_score + time_penalty + length_penalty
        
    if any(fb.guess == guess for fb in past_feedback):
        game_score = -reward_config.get("repetition_penalty")
        return game_score, game_score + time_penalty + length_penalty

    # Build a correct, consistent clue state from history.
    known_green = {}  # {index: letter}
    # Use a Counter for yellow letters to handle duplicates
    known_yellow = Counter()
    known_gray = set()

    for fb in past_feedback:
        guess_letters = list(fb.guess)
        feedback_chars = fb.feedback.split()
        
        # This counter tracks letters that are green or yellow in this specific turn
        counts_in_secret_this_turn = Counter()

        # Identify all Green and Yellow letters
        for i in range(5):
            letter = guess_letters[i]
            if feedback_chars[i] in ('G', 'Y'):
                counts_in_secret_this_turn[letter] += 1
        
        # Update the global minimum required count for yellow letters
        for letter, count in counts_in_secret_this_turn.items():
            known_yellow[letter] = max(known_yellow[letter], count)

        # Process all clues to update the global state
        for i in range(5):
            letter = guess_letters[i]
            feedback = feedback_chars[i]

            if feedback == 'G':
                known_green[i] = letter
            elif feedback == 'X':
                # A letter is only truly gray if it wasn't found as Green or Yellow
                # anywhere in this guess.
                if counts_in_secret_this_turn[letter] == 0:
                    known_gray.add(letter)

    # A letter that is confirmed Green is the highest truth. It cannot also be
    # considered a yellow or gray constraint.
    green_letters = set(known_green.values())
    for letter in green_letters:
        if letter in known_yellow:
            del known_yellow[letter] # No longer a 'yellow' constraint
        if letter in known_gray:
            known_gray.remove(letter) # Cannot be gray if it's green

    # Calculate violations based on the clean, final clue state.
    green_violations = 0
    for idx, correct_letter in known_green.items():
        if guess[idx] != correct_letter:
            green_violations += 1

    yellow_violations = 0
    guess_counts = Counter(guess)
    # Check if the guess has at least the required number of each yellow letter
    for yellow_letter, required_count in known_yellow.items():
        if guess_counts[yellow_letter] < required_count:
            yellow_violations += 1
            
    gray_violations = 0
    # Iterate over unique letters to avoid double-penalizing
    for letter_in_guess in set(guess):
        if letter_in_guess in known_gray:
            gray_violations += 1

    # 4. Calculate total penalty from violations.
    total_penalty = 0.0
    total_penalty += green_violations * reward_config.get("green_position_penalty")
    total_penalty += yellow_violations * reward_config.get("yellow_letter_penalty")
    total_penalty += gray_violations * reward_config.get("gray_letter_penalty")

    # 5. Add soft penalty for out-of-dictionary words.
    if not is_valid_guess(guess, allowed_words):
        total_penalty += reward_config.get("not_in_dictionary_penalty", 25.0)

    # 6. Calculate potential score.
    stagnation_penalty = calculate_stagnation_penalty(guess, known_green, known_yellow, config)
    total_penalty += stagnation_penalty
    strategic_bonus = get_strategic_bonus(guess, past_feedback, config)
    potential_score = reward_config.get("valid_guess_base") + strategic_bonus
    
    current_feedback = get_feedback(guess, secret_word)
    reduction_bonus = reward_for_possibility_reduction(past_feedback, current_feedback, config)
    potential_score += reduction_bonus

    # 7. Calculate final scores.
    game_score = potential_score - total_penalty
    training_reward = game_score + time_penalty + length_penalty
    
    if print_debug:
        print(f"--- REWARD DEBUG ---")
        print(f"  Guess: {guess}, Secret: {secret_word}")
        print(f"  Violations (G/Y/X): {green_violations}/{yellow_violations}/{gray_violations}")
        print(f"  Final Game Score: {game_score:.2f}")
        print(f"--------------------")
        
    return game_score, training_reward


