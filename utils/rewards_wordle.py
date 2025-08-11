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
from utils import prompt
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

# Pre-calculate letter frequencies based on the solution words for accuracy.
# This only runs once when the script starts.
from collections import Counter
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
    reward: float
    feedback_given: GuessFeedback
    
@dataclass
class GameRollout:
    """Contains the full history and outcome of one played game."""
    attempts: List[GenerationAttempt] = field(default_factory=list)
    secret_word: str = ""
    solved: bool = False

def get_feedback(guess: str, secret_word: str) -> GuessFeedback:
    guess = guess.upper()
    secret_word = secret_word.upper()
    if len(guess) != 5:
        return GuessFeedback(guess, "Invalid length")
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

def reward_for_format(completion: str, config: cfg.TrainerConfig) -> float:
    try:
        if not ("<think>" in completion and "</think>" in completion and "<guess>" in completion and "</guess>" in completion): return 0.0
        guess = parse_guess(completion)
        if not guess: return 0.0
        reward = config.reward["format_base"]
        if len(guess) == 5: reward += config.reward["format_len_bonus"]
        return reward
    except Exception: return 0.0

def reward_for_feedback_use(completion: str, history_str: str, config: cfg.TrainerConfig) -> float:
    try:
        guess = parse_guess(completion)
        if not guess or len(guess) != 5: return 0.0
        history_list = ast.literal_eval(history_str) if history_str else []
        
        if not history_list: 
            # return 1 for the guess we dont want to penalize on the first guess
            # avoid reward hacking?
            return 1.0
        
        correct, valid, wrong = {}, {}, set()
        for _, feedback_str in history_list:
            for i, fb in enumerate(feedback_str.split(" ")):
                letter, status = fb[0], fb[2]
                if status == '✓': correct.setdefault(letter, set()).add(i)
                elif status == '-': valid.setdefault(letter, set()).add(i)
                else: wrong.add(letter)
        
        reward = 0.0
        for idx, letter in enumerate(guess):
            if letter in correct and idx in correct[letter]:
                reward += config.reward["feedback_correct_pos"]
            elif letter in valid and idx not in valid.get(letter, set()):
                reward += config.reward["feedback_correct_letter"]
            elif letter in valid and idx in valid.get(letter, set()):
                reward += config.reward["feedback_reuse_penalty"]
            elif letter in wrong:
                reward += config.reward["feedback_wrong_letter_penalty"]
            else:
                reward += config.reward["feedback_new_letter"]
        return max(0, reward)
    except Exception: 
        return 0.0

def reward_partial_credit(guess: str, secret_word: str, config: cfg.TrainerConfig) -> float:
    if not guess or not secret_word or len(guess) != 5: return 0.0
    guess, secret_word = guess.upper(), secret_word.upper()
    reward = config.reward["partial_base"]
    secret_counts = Counter(secret_word)
    scored = [False] * len(guess)

    for i in range(len(guess)):
        if guess[i] == secret_word[i]:
            reward += config.reward["partial_green"]
            secret_counts[guess[i]] -= 1
            scored[i] = True

    for i in range(len(guess)):
        if not scored[i]:
            if guess[i] in secret_counts and secret_counts[guess[i]] > 0:
                reward += config.reward["partial_yellow"]
                secret_counts[guess[i]] -= 1
    return reward


def is_valid_guess(guess, allowed_words):
    return (
        guess
        and len(guess) == 5
        and guess.isalpha()
        and guess.upper() in {w.upper() for w in allowed_words}
    )

# def reward_for_entropy_proxy(guess: str, config: cfg.TrainerConfig) -> float:
#     """
#     Calculates a reward bonus based on heuristics that approximate high-entropy
#     guesses: unique letters and common letters.
#     """
#     if not guess or len(guess) != 5:
#         return 0.0

#     # Bonus for number of unique letters
#     unique_letters = set(guess)
#     unique_bonus = (len(unique_letters) / 5.0) * config.reward.get("entropy_unique_letter_bonus")

#     # Bonus for using common letters, weighted by their frequency
#     common_letter_score = sum(NORMALIZED_LETTER_FREQS.get(letter, 0) for letter in unique_letters)
#     # Normalize by a typical score for a 5-unique-letter word to keep the bonus in a stable range
#     common_bonus = (common_letter_score / 3.5) * config.reward.get("entropy_common_letter_bonus")
    
#     return unique_bonus + common_bonus

def calculate_total_reward(
    response: str,
    secret_word: str,
    past_feedback: List[GuessFeedback],
    config: cfg.TrainerConfig,
    allowed_words: set
) -> float:
    """
    A reward function focused on teaching the model the most important rules of Wordle.
    """
    guess = parse_guess(response)

    # Catastrophic failure: The model didn't produce a parsable guess at all.
    if not guess:
        return config.reward.get("format_fail_penalty")

    # --- Step 1 (THE MOST IMPORTANT): Check for a win immediately. ---
    # This is the highest possible reward and should override all other checks.
    if guess == secret_word.upper():
        return config.reward.get("solution_correct_guess")

    # --- Step 2: Now, check for other failures like invalid words or repeats. ---
    if not is_valid_guess(guess, allowed_words):
        return config.reward.get("format_fail_penalty")

    if any(fb.guess == guess for fb in past_feedback):
        return config.reward.get("repetition_penalty")

    # --- Step 3: If it's a valid, non-winning, non-repeated guess, evaluate it. ---
    # (The rest of the function for clue violation penalties remains the same)
    
    known_green = {}
    known_yellow = set()
    all_valid_guessed_letters = set()

    for fb in past_feedback:
        feedback_parts = fb.feedback.split()
        if len(fb.guess) == 5 and len(feedback_parts) == 5:
            all_valid_guessed_letters.update(fb.guess)
            for i, f_char in enumerate(feedback_parts):
                letter = fb.guess[i]
                if f_char == '✓':
                    known_green[i] = letter
                elif f_char == '-':
                    if i not in known_green or known_green[i] != letter:
                         known_yellow.add(letter)

    known_gray = all_valid_guessed_letters - set(known_green.values()) - known_yellow
    
    violated_clue_penalty = config.reward.get("violated_clue_penalty")

    if any(letter in known_gray for letter in guess):
        return violated_clue_penalty

    if not known_yellow.issubset(set(guess)):
        return violated_clue_penalty

    for idx, letter in known_green.items():
        if guess[idx] != letter:
            return violated_clue_penalty
            
    # --- Step 4: If no rules were broken, give a small neutral reward. ---
    return config.reward.get("valid_guess_base")


def format_prompt_for_model(past_feedback: List[GuessFeedback], system_prompt: str) -> List[dict]:
    """
    Formats the history of guesses into a message list for the model.
    This version provides standard feedback for all guesses and adds a note
    for words not in the dictionary.
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

        # For all other guesses (in or out of dictionary), format the standard feedback
        feedback_parts = fb.feedback.split()
        feedback_str = " ".join([f"{fb.guess[j]}({f})" for j, f in enumerate(feedback_parts)])
        line = f"* Guess {i+1}: {fb.guess} \u2192 {feedback_str}"
        
        # Add a special note if the word was not in the dictionary
        if not fb.is_in_dictionary:
            line += "  *(Note: This word is not in the dictionary.)*"
            
        prompt_parts.append(line)

    user_content = "\n".join(prompt_parts)
    print("DEBUG: user content:\n", user_content)
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
                # This is simpler and more reliable than parsing feedback from a string.
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
    initial_history: list = None
) -> GameRollout:
    """
    Plays a game of Wordle using a self-contained reward system.
    This version contains fixes for more precise and helpful feedback.
    """
    game_rollout = GameRollout(secret_word=secret_word)
    past_feedback: List[GuessFeedback] = []
    already_guessed_words = set()
    max_trials = config.rl.max_trials
    num_generations = config.rl.num_generations
    print("DEBUG: Starting game with secret word:", secret_word)

    if initial_history:
        past_feedback = _reconstruct_past_feedback(initial_history, secret_word)
        for fb in past_feedback:
            # Only add valid past guesses to the already_guessed_words set
            if fb.guess not in ["INVALID_FORMAT", "REPEATED_GUESS", "WORD_NOT_IN_LIST"]:
                 already_guessed_words.add(fb.guess)
        if any(fb.guess == secret_word.upper() for fb in past_feedback):
             return game_rollout

    for attempt_num in range(len(past_feedback), max_trials):
        messages = format_prompt_for_model(past_feedback, system_prompt)
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
        for response in generations:
            response_only = response.replace(prompt_string, "").strip()
            guess = parse_guess(response_only)

            # The reward function now checks for the win condition first.
            reward = calculate_total_reward(
                response_only, secret_word, past_feedback, config, ALLOWED_GUESSES
            )

            attempt = GenerationAttempt(
                prompt_string=prompt_string,
                prompt_tokens=prompt_tokens,
                full_response=response_only,
                response_tokens=tokenizer.encode(response_only),
                parsed_guess=guess, # This will be the word string or None
                reward=reward,
                feedback_given=None
            )
            current_turn_attempts.append(attempt)
            
        # Filter for only valid, new, parsable attempts to advance the game state.
        valid_candidates = [
            att for att in current_turn_attempts
            if att.parsed_guess and att.parsed_guess not in already_guessed_words
        ]
          # If NO valid, new candidates were generated in this turn, the game is stuck.
        if not valid_candidates:
            print("No valid, new guesses were generated this turn. Ending game early.")
            # Still add all the failed attempts for the trainer to learn from.
            game_rollout.attempts.extend(current_turn_attempts)
            break

        # 4. Select the best candidate from the VALID ones to advance the game state
        best_attempt = max(valid_candidates, key=lambda att: att.reward)
        best_guess = best_attempt.parsed_guess

        # 5. Generate and append feedback for the chosen path
        is_valid_in_dict = is_valid_guess(best_guess, ALLOWED_GUESSES)
        feedback = get_feedback(best_guess, secret_word)
        feedback.is_in_dictionary = is_valid_in_dict
        
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
            print(f"🎉 SUCCESS on attempt {attempt_num + 1}! Word: '{secret_word.upper()}'")
            game_rollout.solved = True
            break
            
    if not game_rollout.solved:
        valid_guesses_in_game = [
            fb.guess for fb in past_feedback 
            if fb.guess not in ['INVALID_FORMAT', 'REPEATED_GUESS']
        ]
        print(f"❌ Did not guess '{secret_word.upper()}' in {max_trials} trials. Guesses: {valid_guesses_in_game}")
        
    return game_rollout