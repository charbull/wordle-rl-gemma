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
file_url_possible_answers = "https://raw.githubusercontent.com/Roy-Orbison/wordle-guesses-answers/refs/heads/main/answers.txt"

# --- Word Lists ---
SOLUTION_WORDS = read_files.load_word_list_from_url(file_url_possible_answers, "./data/nyt_possible_wordle_list.txt")
GUESS_TAG_RE = re.compile(r"<guess>(.*?)</guess>", re.DOTALL | re.IGNORECASE)
# This should only match 5 ALL-CAPS letters
FIVE_LETTER_WORD_RE = re.compile(r'\b([A-Z]{5})\b')

@dataclass
class GuessFeedback:
    """Represents a single guess and its corresponding feedback."""
    guess: str
    feedback: str

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

def calculate_total_reward(
    response: str,
    secret_word: str,
    past_feedback: List[GuessFeedback],
    config: cfg.TrainerConfig,
    allowed_words: set
) -> float:
    """
    A simplified and stricter reward function focused on teaching the model
    the most important rules of Wordle.
    """
    # --- Step 1: Handle catastrophic failures (bad format, repetition) ---
    guess = parse_guess(response)
    if not guess or not is_valid_guess(guess, allowed_words):
        return config.reward.get("format_fail_penalty")

    if any(fb.guess == guess for fb in past_feedback):
        return config.reward.get("repetition_penalty")

    # --- Step 2: Check for a win (the highest possible reward) ---
    if guess == secret_word.upper():
        return config.reward.get("solution_correct_guess")

    # --- Step 3: Build knowledge from past clues ---
    known_green = {}  # {index: letter}
    known_yellow = set()
    all_valid_guessed_letters = set()

    # Iterate through past feedback and only process valid, 5-letter guess entries
    for fb in past_feedback:
        feedback_parts = fb.feedback.split()
        if len(fb.guess) == 5 and len(feedback_parts) == 5:
            all_valid_guessed_letters.update(fb.guess)
            for i, f_char in enumerate(feedback_parts):
                letter = fb.guess[i]
                if f_char == '✓':
                    known_green[i] = letter
                elif f_char == '-':
                    # Add to yellow only if not already confirmed green in that spot
                    if i not in known_green or known_green[i] != letter:
                         known_yellow.add(letter)

    # Now, calculate gray letters from the set of letters from valid guesses
    known_gray = all_valid_guessed_letters - set(known_green.values()) - known_yellow
    # --- Step 4: Apply a large penalty for violating any known clue ---
    # This is the most important rule for the model to learn.
    violated_clue_penalty = config.reward.get("violated_clue_penalty")

    # Violated a gray letter clue (used a letter that's known to be absent)
    if any(letter in known_gray for letter in guess):
        return violated_clue_penalty

    # Violated a green letter clue (didn't use a green letter in the right spot)
    for idx, letter in known_green.items():
        if guess[idx] != letter:
            return violated_clue_penalty

    # Violated a yellow letter clue (didn't use a required yellow letter)
    if not known_yellow.issubset(set(guess)):
        return violated_clue_penalty

    # --- Step 5: If no rules were broken, give a small neutral reward ---
    # This encourages exploration without rewarding suboptimal guesses.
    # It tells the model "Okay, that was a valid move, now try to win."
    return config.reward.get("valid_guess_base")



def format_prompt_for_model(past_feedback: List[GuessFeedback], system_prompt: str) -> List[dict]:
    """Formats the history of guesses into a message list for the model."""
    if not past_feedback:
        user_content = "This is the first turn. Please provide your best starting word."
    else:
        prompt_parts = ["**Clues so far:**"]
        for i, fb in enumerate(past_feedback):
            if fb.guess in ["INVALID_FORMAT", "INVALID_LENGTH", "REPEATED_GUESS", "WORD_NOT_IN_LIST"]:
                prompt_parts.append(f"* Attempt {i+1} Error: {fb.feedback}")
            else:
                feedback_str = " ".join([f"{fb.guess[j]}({f})" for j, f in enumerate(fb.feedback.split())])
                prompt_parts.append(f"* Guess {i+1}: {fb.guess} \u2192 {feedback_str}")
        user_content = "\n".join(prompt_parts)
        print("debug:: user content: ", user_content)
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
    Plays a game of Wordle using a self-contained reward system that does not
    require an external valid word list. This version includes heuristic
    re-ranking to make smarter guess selections.
    """
    game_rollout = GameRollout(secret_word=secret_word)
    past_feedback: List[GuessFeedback] = []
    already_guessed_words = set() #
    # Maximum number of attempts allowed in a game
    max_trials = config.rl.max_trials
    # Number of parallel generations to sample per turn
    num_generations = config.rl.num_generations
    total_rewards = []

    if initial_history:
        # Use the helper to convert the dataset's message history into our internal format
        past_feedback = _reconstruct_past_feedback(initial_history, secret_word)
        
        # Populate the set of already guessed words from the reconstructed history
        for fb in past_feedback:
            already_guessed_words.add(fb.guess)

        # OPTIONAL: Check if the game was already solved in the provided history
        if any(fb.guess == secret_word.upper() for fb in past_feedback):
             print(f"INFO: Game for '{secret_word}' already solved in initial_history. Skipping RL rollout.")
             # Returning here means we won't generate new attempts for already-solved games.
             # This is efficient as there's nothing new to learn.
             return game_rollout


    for attempt_num in range(len(past_feedback), max_trials):
        # 1. Format the prompt based on the history of previous turns
        messages = format_prompt_for_model(past_feedback, system_prompt)

        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_string)

        # 2. Generate N parallel responses for the current state
        generations = [
            generate(
                model, tokenizer, prompt=prompt_string,
                max_tokens=config.rl.max_completion_length, sampler=sampler, verbose=False
            ).replace(prompt.SYSTEM_PROMPT, "").strip() for _ in range(num_generations)
        ]

        # 3. Calculate rewards for each generation and create attempt objects
        current_turn_attempts: List[GenerationAttempt] = []
        for i, response in enumerate(generations):
            response = response.replace(prompt_string, "").strip()
            print("debug model response: ", response)
            print("secret word is: ", secret_word)
            reward = calculate_total_reward(
                response,
                secret_word,
                past_feedback,
                config,
                SOLUTION_WORDS
            ) 
            # length_penalty_per_token = config.reward.get("length_penalty_per_token")
            # num_tokens = len(tokenizer.encode(response))
            # reward = main_reward - (length_penalty_per_token * num_tokens)
            total_rewards.append(reward)
            
            guess = parse_guess(response)
            print(f"debug: parsed guess: {guess}")
            is_repeat = guess and (guess in already_guessed_words)
            if not guess:
                parsed_guess_state = "INVALID_FORMAT"

            elif not is_valid_guess(guess, SOLUTION_WORDS):
                parsed_guess_state = "WORD_NOT_IN_LIST"

            elif is_repeat:
                parsed_guess_state = "REPEATED_GUESS"
            else:
                parsed_guess_state = guess

            attempt = GenerationAttempt(
                prompt_string=prompt_string,
                prompt_tokens=prompt_tokens,
                full_response=response,
                response_tokens=tokenizer.encode(response),
                parsed_guess=parsed_guess_state,
                reward=reward,
                feedback_given=None  # To be filled in after the best attempt is chosen
            )
            current_turn_attempts.append(attempt)

        all_invalid = all(
                att.parsed_guess in ["INVALID_FORMAT", "REPEATED_GUESS"]
                for att in current_turn_attempts
            )
        if all_invalid:
                print("All attempts invalid this turn. Ending game early.")
                break
        # Select the generation with the highest reward. The heuristic bonus has been removed
        # to align the training objective with the action selection mechanism.
        best_generation_idx = -1
        max_reward = -float('inf')
        
        for i, attempt in enumerate(current_turn_attempts):
            if attempt.reward > max_reward:
                max_reward = attempt.reward
                best_generation_idx = i

        # 4. Use the best generation (chosen by max reward) to advance the game state
        best_attempt = current_turn_attempts[best_generation_idx]
        best_guess = best_attempt.parsed_guess

        # Add the played word to the set to check for future repetitions
        if best_guess not in ["INVALID_FORMAT", "REPEATED_GUESS"]:
            already_guessed_words.add(best_guess)

        # 5. Generate specific feedback for the model's *next* turn based on the best attempt
        if best_guess == "INVALID_FORMAT":
            feedback = GuessFeedback(
                "INVALID_FORMAT", "Your response did not contain a valid 5-letter guess.")
        elif best_guess == "REPEATED_GUESS":
            feedback = GuessFeedback(
                "REPEATED_GUESS", "You have already guessed that word.")
        elif best_guess == "WORD_NOT_IN_LIST":
            feedback = GuessFeedback(
                "WORD_NOT_IN_LIST", "That word is not in the official Wordle list.")
        else:
            # The guess is a 5-letter word, get its official Wordle feedback
            feedback = get_feedback(best_guess, secret_word)

        # Store the results of this turn
        best_attempt.feedback_given = feedback
        game_rollout.attempts.extend(current_turn_attempts)
        past_feedback.append(feedback)

        # 6. Check for game termination
        if best_guess == secret_word.upper():
            print(
                f"🎉 SUCCESS on attempt {attempt_num + 1}! Word: '{secret_word.upper()}'")
            game_rollout.solved = True
            break

    if not game_rollout.solved:
        print(
            f"❌ Did not guess '{secret_word.upper()}' in {max_trials} trials. guesses: {[fb.guess for fb in past_feedback]}")
    return game_rollout