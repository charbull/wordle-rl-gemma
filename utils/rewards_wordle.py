### rewards functions that are specific to a given dataset """
# TODO make a class and have it inherit from a base class and implement the methods
# We need the following functions:
# 1) `format_prompt_from_dataset(sample: dict) -> str`:
# This function should take a sample from the dataset and return a formatted prompt string.
# 2) `parse_guess(response: str) -> str`:
# This function should parse the response from the model and return the guess. It is used for logging
# 3) total_reward(completion: str, sample: dict, config: cfg.TrainerConfig) -> float:
# This function should calculate the total reward for a given completion based on the sample and configuration.
#######

import config as cfg
from collections import Counter
import re
import ast
from typing import List
from mlx_lm import generate
from dataclasses import dataclass, field
from utils import read_files
from utils import config as cfg
# ==============================================================================
# ---  DATA STRUCTURES FOR GAME LOGIC ---
# ==============================================================================
URL_POSSIBLE_ANSWERS = "https://gist.githubusercontent.com/kcwhite/bb598f1b3017b5477cb818c9b086a5d9/raw/5a0adbbb9830ed93a573cb87a7c14bb5dd0b1883/wordle_possibles.txt"
LOCAL_FILE_PATH = "./data/nyt_possible_wordle_list.txt"
_ALLOWED_GUESSES_CACHE = None
# --- Word Lists ---
ALLOWED_GUESSES = read_files.load_word_list_from_url(URL_POSSIBLE_ANSWERS, LOCAL_FILE_PATH)
GUESS_TAG_RE = re.compile(r"<guess>(.*?)</guess>", re.DOTALL | re.IGNORECASE)
# This should only match 5 ALL-CAPS letters
FIVE_LETTER_WORD_RE = re.compile(r'\b([A-Z]{5})\b')

#============================================================================= 
# --- Reward Calculation Helpers ---
#=============================================================================
# Pre-calculate letter frequencies based on the solution words for accuracy.
# This only runs once when the script starts.
# from collections import Counter
LETTER_FREQUENCIES = Counter(
    letter for word in ALLOWED_GUESSES for letter in word.upper()
)
MAX_FREQ = max(LETTER_FREQUENCIES.values())
NORMALIZED_LETTER_FREQS = {
    letter: freq / MAX_FREQ for letter, freq in LETTER_FREQUENCIES.items()
}

def get_allowed_guesses():
    """
    Loads the list of allowed guess words, caching the result after the first call.
    This prevents file I/O from running on simple module import.
    """
    global _ALLOWED_GUESSES_CACHE
    if _ALLOWED_GUESSES_CACHE is None:
        print("--- Loading Wordle guess list for the first time... ---")
        _ALLOWED_GUESSES_CACHE = read_files.load_word_list_from_url(
            URL_POSSIBLE_ANSWERS, LOCAL_FILE_PATH
        )
    return _ALLOWED_GUESSES_CACHE

def get_letter_frequencies():
    """Calculates and caches letter frequencies based on the solution words."""
    global NORMALIZED_LETTER_FREQS # Assuming this is the name you want
    # Check if it's already calculated, if not, compute it
    if "NORMALIZED_LETTER_FREQS" not in globals():
        from collections import Counter
        allowed_words = get_allowed_guesses()
        letter_counts = Counter(letter for word in allowed_words for letter in word.upper())
        max_freq = max(letter_counts.values()) if letter_counts else 1
        # Declare as global to cache it
        globals()["NORMALIZED_LETTER_FREQS"] = {
            letter: freq / max_freq for letter, freq in letter_counts.items()
        }
    return globals()["NORMALIZED_LETTER_FREQS"]

def reward_for_entropy_proxy(guess: str, config: cfg.TrainerConfig) -> float:
    """
    Calculates a reward bonus based on heuristics that approximate high-entropy
    guesses: unique letters and common letters.
    """
    if not guess or len(guess) != 5:
        return 0.0

    # Bonus for number of unique letters
    unique_letters = set(guess)
    unique_bonus = (len(unique_letters) / 5.0) * config.reward.get("entropy_unique_letter_bonus")

    # Bonus for using common letters, weighted by their frequency
    common_letter_score = sum(NORMALIZED_LETTER_FREQS.get(letter, 0) for letter in unique_letters)
    # Normalize by a typical score for a 5-unique-letter word to keep the bonus in a stable range
    common_bonus = (common_letter_score / 3.5) * config.reward.get("entropy_common_letter_bonus")
    
    return unique_bonus + common_bonus

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

def get_feedback(guess: str, secret_word: str) -> GuessFeedback:
    """
    Given a guess and the secret word, returns the feedback string in the format:
    '✓' for correct position, '-' for correct letter wrong position, 'x' for incorrect letter.
    Example: guess='CRANE', secret_word='CLEAN' -> feedback='✓ - ✓ x -'
    """
    guess = guess.upper()
    secret_word = secret_word.upper()
    if len(guess) != 5:
        return GuessFeedback(guess, "INVALID_FORMAT")
    feedback = [''] * 5
    secret_counts = Counter(secret_word)
    for i in range(5):
        if guess[i] == secret_word[i]:
            feedback[i] = '✓'
            secret_counts[guess[i]] -= 1
    for i in range(5):
        if feedback[i] == '':
            if guess[i] in secret_counts and secret_counts[guess[i]] > 0:
                feedback[i] = '-'
                secret_counts[guess[i]] -= 1
            else:
                feedback[i] = 'x'
    return GuessFeedback(guess, " ".join(feedback))

def format_prompt_from_dataset(sample):
    history_str = sample['past_guess_history']
    try:
        history_list = ast.literal_eval(history_str)
    except (ValueError, SyntaxError):
        history_list = []
    if not history_list:
        return "This is the first turn. Please provide your best starting word."
    prompt_parts = ["**Clues so far:**"]
    for i, (guess, feedback) in enumerate(history_list):
        prompt_parts.append(f"* Guess {i+1}: {guess} → {feedback}")
    return "\n".join(prompt_parts)

def parse_guess(response: str) -> str:
    """
    A parser that finds the last 5-letter ALL CAPS word inside the <guess> tags.
    This prevents the model from being rewarded for "thinking out loud" inside the tags.
    """
    match = GUESS_TAG_RE.search(response)
    if not match:
        return None

    # Get the original content without changing its case.
    content_inside_tags = match.group(1)
    
    # Use the regex to find all 5-letter ALL CAPS words.
    five_letter_words = FIVE_LETTER_WORD_RE.findall(content_inside_tags)

    # If any were found, return the last one.
    if five_letter_words:
        # The result is already uppercase, so just return it.
        return five_letter_words[-1]
    
    # Otherwise, no validly formatted guess was found.
    return None

def is_valid_guess(guess, allowed_words):
    """
    Checks if the guess is a valid 5-letter word in the allowed words list.
    """
    return (
        guess
        and len(guess) == 5
        and guess.isalpha()
        and guess.upper() in {w.upper() for w in allowed_words}
    )

def calculate_total_reward(
    response: str,
    secret_word: str,
    past_feedback: List[GuessFeedback],
    config: cfg.TrainerConfig,
    allowed_words: set,
    tokenizer = None
) -> tuple[float, float]: # Return a tuple of (game_score, training_reward)
    """
    Calculates a training_reward and a game score for a given model response.
    - Prioritizes winning.
    - Penalizes invalid/repeated guesses.
    - Applies GRADUATED penalties for violating known clues.
    - Rewards valid, consistent guesses with a base reward and an entropy bonus.
    - Applies a small time penalty to every guess to encourage efficiency.
    """
    reward_config = config.reward
    guess = parse_guess(response)

    # These apply to the training_reward but not the game_score.
    time_penalty = reward_config.get("time_penalty_per_guess", -1.0)
    length_penalty = len(tokenizer.encode(response)) * reward_config.get("length_penalty_per_token", -0.01)

    game_score = 0.0

    # PRIORITY 1: Handle parsing and win condition
    if not guess:
        game_score = reward_config.get("format_fail_penalty")
    elif guess == secret_word.upper():
        game_score = reward_config.get("solution_correct_guess")
        
    # PRIORITY 2: Handle fundamental game rule violations
    elif not is_valid_guess(guess, allowed_words) and \
          reward_config.get("not_in_dictionary_penalty") != 0.0:
        game_score = reward_config.get("not_in_dictionary_penalty")
    elif any(fb.guess == guess for fb in past_feedback):
        game_score = reward_config.get("repetition_penalty")
    else:
        # PRIORITY 3: If the guess is valid, new, and not a win, check for logical clue violations.
        known_green = {}   # {position: letter}
        known_yellow = set() # Set of letters that are yellow
        known_gray = set()   # Set of letters that are gray

        # This loop correctly populates the clue sets.
        for fb in past_feedback:
            for i, f_char in enumerate(fb.feedback.split()):
                letter = fb.guess[i]
                if f_char == '✓':
                    known_green[i] = letter
                elif f_char == '-':
                    known_yellow.add(letter)
                elif f_char == 'x':
                    known_gray.add(letter)
        
        # A letter cannot be both gray and yellow/green.
        # This prevents double-penalizing.
        known_gray -= set(known_green.values())
        known_gray -= known_yellow

        total_penalty = 0.0
        
        # 1. Calculate penalty for each gray letter used.
        gray_violations = sum(1 for letter in guess if letter in known_gray)
        total_penalty += gray_violations * reward_config.get("gray_letter_penalty", -5.0)

        # 2. Calculate penalty for each yellow letter NOT used.
        yellow_violations = sum(1 for letter in known_yellow if letter not in guess)
        total_penalty += yellow_violations * reward_config.get("yellow_letter_penalty", -10.0)

        # 3. Calculate penalty for each green letter in the wrong position.
        green_violations = sum(1 for idx, letter in known_green.items() if guess[idx] != letter)
        total_penalty += green_violations * reward_config.get("green_position_penalty", -15.0)

        if total_penalty < 0:
            game_score = total_penalty
        else:
            # PRIORITY 4: This is a "good" guess that follows all rules. Reward it.
            base_reward = reward_config.get("valid_guess_base")
            entropy_bonus = reward_for_entropy_proxy(guess, config)
            game_score = base_reward + entropy_bonus

    # Finally, calculate the full training_reward
    training_reward = game_score + time_penalty + length_penalty
    
    return game_score, training_reward


def format_prompt_for_model(past_feedback: List[GuessFeedback], system_prompt: str) -> List[dict]:
    """
    Formats the history of guesses into a message list for the model.
    """
    if not past_feedback:
        user_content = "This is the first turn. Please provide your best starting word."
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    prompt_parts = ["**Clues so far:**"]
    
    for i, fb in enumerate(past_feedback):
        # Handle hard errors like invalid format or repeated words first
        if fb.guess in ["INVALID_FORMAT", "REPEATED_GUESS"]:
            prompt_parts.append(f"* Attempt {i+1} Error: {fb.feedback}")
            continue

        feedback_parts = fb.feedback.split()
        descriptions = []
        correct_pos = [f"{fb.guess[j]}" for j, f in enumerate(feedback_parts) if f == '✓']
        wrong_pos = [f"{fb.guess[j]}" for j, f in enumerate(feedback_parts) if f == '-']
        not_in_word = [f"{fb.guess[j]}" for j, f in enumerate(feedback_parts) if f == 'x']

        if correct_pos:
            descriptions.append(f"{', '.join(correct_pos)} are in the correct position.")
        if wrong_pos:
            descriptions.append(f"{', '.join(wrong_pos)} are in the word, but in the wrong position.")
        if not_in_word:
            descriptions.append(f"{', '.join(not_in_word)} are not in the word.")

        feedback_str = " ".join(descriptions) if descriptions else "No information gained."
        line = f"* Guess {i+1}: {fb.guess} → {feedback_str}"
        prompt_parts.append(line)
    
    user_content = "\n".join(prompt_parts)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

def _reconstruct_past_feedback(messages: list, secret_word: str) -> List[GuessFeedback]:
    """
    Parses a 'messages' list from a dataset trajectory and reconstructs the
    internal List[GuessFeedback] format by re-calculating feedback for each
    past guess against the secret word.

    Args:
        messages: The list of message dictionaries from the dataset.
        secret_word: The secret word for the current puzzle.

    Returns:
        A list of GuessFeedback objects representing the game's history.
    """
    past_feedback = []
    # Find all past guesses from 'assistant' messages in the history
    for msg in messages:
        if msg.get("role") == "assistant":
            # Use your existing robust parser to find the guess
            guess = parse_guess(msg.get("content", ""))
            if guess:
                # Re-calculate the feedback to ensure it's in the correct format.
                feedback = get_feedback(guess, secret_word)
                past_feedback.append(feedback)
    return past_feedback

def play_wordle_game(
    model,
    tokenizer,
    secret_word: str,
    system_prompt: str,
    config: cfg.TrainerConfig,
    sampler,
    initial_history: list = None,
    print_debug = False
) -> GameRollout:
    """
    Plays a game of Wordle with rl.num_generations, calculate the total rewards for each generation,
    and returns the full game rollout including all attempts and the final outcome.

    Args:
        model: The language model to use for generating guesses.
        tokenizer: The tokenizer corresponding to the model.
        secret_word: The secret word to be guessed in this game.
        system_prompt: The system prompt to guide the model's behavior.
        config: Configuration parameters for the RL training and game settings.
        sampler: The sampling strategy to use for generation (e.g., temperature, top-k).
        initial_history: Optional initial history of past guesses and feedback to continue from.

    Returns:
        A GameRollout object containing the full history of attempts and whether the game was solved.
    """
    game_rollout = GameRollout(secret_word=secret_word)
    past_feedback: List[GuessFeedback] = []
    already_guessed_words = set()
    max_trials = config.rl.max_trials
    num_generations = config.rl.num_generations
    if print_debug:
        print(f"\n{'='*35}\n|| NEW GAME || Secret Word: {secret_word.upper()}\n{'='*35}")
    
    # This is provided by the dataset for continuing games.
    if initial_history:
        past_feedback = _reconstruct_past_feedback(initial_history, secret_word)
        for fb in past_feedback:
            # Only add valid past guesses to the already_guessed_words set
            if fb.guess not in ["INVALID_FORMAT", "REPEATED_GUESS", "WORD_NOT_IN_LIST"]:
                 already_guessed_words.add(fb.guess)
        if any(fb.guess == secret_word.upper() for fb in past_feedback):
             return game_rollout

    # Main game loop start from initial history length to max trials
    for attempt_num in range(len(past_feedback), max_trials):
        if print_debug:
            print(f"\n--- Turn {attempt_num + 1}/{max_trials} ---")
        messages = format_prompt_for_model(past_feedback, system_prompt)
        if print_debug:
            print(f"Prompt sent to model:\n{messages[-1]['content']}")
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_string)

        # Generate N parallel responses from the current state of the model.
        generations = [
            generate(
                model, tokenizer, prompt=prompt_string,
                max_tokens=config.rl.max_completion_length, sampler=sampler, verbose=False
            ) for _ in range(num_generations)
        ]

        # Process each generation to extract the guess and calculate the reward.
        current_turn_attempts: List[GenerationAttempt] = []
        for i, response in enumerate(generations):
            response_only = response.replace(prompt_string, "").strip()
            guess = parse_guess(response_only)

            # The reward function checks for the win condition first.
            game_score, training_reward = calculate_total_reward(
                response_only, secret_word, past_feedback, config, ALLOWED_GUESSES, tokenizer
            )
            if print_debug:
                print(f"  [Generation {i+1}/{num_generations}]")
                print(f"    Raw Response: \"{response_only}\"")
                print(f"    Parsed Guess: '{guess}'")
                print(f"    Game Score: {game_score:.2f}")
                print(f"    Training Reward: {training_reward:.2f}")

            attempt = GenerationAttempt(
                prompt_string=prompt_string,
                prompt_tokens=prompt_tokens,
                full_response=response_only,
                response_tokens=tokenizer.encode(response_only),
                parsed_guess=guess,
                game_score=game_score,
                training_reward=training_reward,
                feedback_given=None
            )
            current_turn_attempts.append(attempt)
            
        # Filter for only valid, new, parsable attempts to advance the game state.
        # and from the dictionary
        valid_candidates = [
            att for att in current_turn_attempts
            if att.parsed_guess and \
            att.parsed_guess not in already_guessed_words
            # TODO: Consider if we want to penalize words not in the dictionary to make it more strict
            # and \
            #is_valid_guess(att.parsed_guess, ALLOWED_GUESSES)
        ]
          # If NO valid, new candidates were generated in this turn, the game is stuck.
        if not valid_candidates:
            print("No valid, new guesses were generated this turn. Ending game early.")
            # Still add all the failed attempts for the trainer to learn from.
            game_rollout.attempts.extend(current_turn_attempts)
            break

        # Select the best candidate from the VALID ones to advance the game state
        best_attempt = max(valid_candidates, key=lambda att: att.game_score)
        best_guess = best_attempt.parsed_guess

        # Generate and append feedback for the chosen path
        is_valid_in_dict = is_valid_guess(best_guess, ALLOWED_GUESSES)
        feedback = get_feedback(best_guess, secret_word)
        feedback.is_in_dictionary = is_valid_in_dict
        
        if print_debug:
            print("\nTurn Decision:")
            print(f"  - Chose '{best_guess}' (Game Score: {best_attempt.game_score:.2f}) to advance the game.")
            print(f"  - Feedback for next turn: {feedback.feedback}")

        # Add the chosen guess to the set of used words for the next turn's check
        already_guessed_words.add(best_guess)

        # Store the results
        best_attempt.feedback_given = feedback
        # Add ALL attempts (including failed ones) to the rollout for the trainer.
        game_rollout.attempts.extend(current_turn_attempts)
        # Add only the feedback from the BEST VALID path to guide the next turn.
        past_feedback.append(feedback)

        # 6. Check for game termination
        if best_guess == secret_word.upper():
            print(f"🎉🎉🎉 SUCCESS on attempt {attempt_num + 1}! Word: '{secret_word.upper()}' 🎉🎉🎉")
            game_rollout.solved = True
            break
            
    if not game_rollout.solved:
        if print_debug:
            print(f"❌ Did not guess '{secret_word.upper()}' in {max_trials} trials. Turn-by-turn breakdown:")

            from collections import defaultdict
            # Group all generated attempts by their prompt, which represents a single turn.
            grouped_by_turn = defaultdict(list)
            for attempt in game_rollout.attempts:
                # We only care about attempts that produced a parsable guess for this summary
                if attempt.parsed_guess:
                    grouped_by_turn[attempt.prompt_string].append(attempt)
            
            # Ensure we process turns in the order they were played
            # We do this by finding the original index of the first attempt for each turn's prompt
            turn_prompts = sorted(
                grouped_by_turn.keys(), 
                key=lambda p: game_rollout.attempts.index(grouped_by_turn[p][0])
            )

            # Iterate through each turn and print the winner/loser summary
            for i, prompt in enumerate(turn_prompts):
                attempts_for_turn = grouped_by_turn[prompt]
                if not attempts_for_turn:
                    continue

                # The "winner" for this turn is the one with the highest game_score.
                # This is the guess that was chosen to advance the game state.
                winner = max(attempts_for_turn, key=lambda att: att.game_score)
                
                # "Losers" are all other generated candidates for this turn.
                losers = [att for att in attempts_for_turn if att is not winner]

                # Format the strings for clean printing
                winner_str = f"'{winner.parsed_guess}' (Score: {winner.game_score:.2f})"
                if losers:
                    losers_str_list = [f"'{l.parsed_guess}' ({l.game_score:.2f})" for l in losers]
                    losers_str = ", ".join(losers_str_list)
                else:
                    losers_str = "None (only one generation)"

                print(f"  - Turn {i+1}: Winner = {winner_str} | Losers = [{losers_str}]")
    return game_rollout