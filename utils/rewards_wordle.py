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
import json
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
# --- Word Lists ---
ALLOWED_GUESSES = read_files.load_word_list_from_url(URL_POSSIBLE_ANSWERS, LOCAL_FILE_PATH)
GUESS_TAG_RE = re.compile(r"<guess>(.*?)</guess>", re.DOTALL | re.IGNORECASE)
# This should only match 5 ALL-CAPS letters
FIVE_LETTER_WORD_RE = re.compile(r'\b([A-Z]{5})\b')




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

from collections import Counter # Make sure this is at the top of your file
def get_feedback(guess: str, secret_word: str) -> GuessFeedback:
    """
    Given a guess and the secret word, returns the feedback string in the format:
    'G' for correct position, 'Y' for correct letter wrong position, 'X' for incorrect letter.
    """
    guess = guess.upper()
    secret_word = secret_word.upper()
    
    if len(guess) != 5:
        return GuessFeedback(guess, "INVALID_FORMAT")

    feedback = [''] * 5
    secret_counts = Counter(secret_word)

    # First pass: Find all correct letters in the correct position (Greens)
    for i in range(5):
        if guess[i] == secret_word[i]:
            feedback[i] = 'G'
            secret_counts[guess[i]] -= 1

    # Second pass: Find correct letters in wrong positions (Yellows) and incorrect letters (Grays)
    for i in range(5):
        if feedback[i] == '':  # Only check letters that weren't green
            if guess[i] in secret_counts and secret_counts[guess[i]] > 0:
                feedback[i] = 'Y'
                secret_counts[guess[i]] -= 1
            else:
                feedback[i] = 'X'
                
    return GuessFeedback(guess, " ".join(feedback))

#============================================================================= 
# --- Reward Enthropy Helpers ---
#=============================================================================
def load_word_entropy():
    try:
        with open('./data/word_entropy.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: word_entropy.json not found.                   !!!")
        print("!!! Run `calculate_word_entropy_mlx.py` to generate it.     !!!")
        print("!!! The information gain bonus will be zero for this run.   !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {}

WORD_ENTROPY_SCORES = load_word_entropy()


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

# We also need to slightly adjust reward_for_information_gain to not check the turn number
def reward_for_information_gain(guess: str, config: cfg.TrainerConfig) -> float:
    """Provides a reward based on pre-calculated information gain (entropy)."""
    if not guess:
        return 0.0
    entropy_score = WORD_ENTROPY_SCORES.get(guess.upper(), 0.0)
    return entropy_score * config.reward.get("information_gain_bonus_coeff")


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
            guess and len(guess) == 5 and guess.isalpha() and
            guess.upper() in allowed_words
        )




def format_prompt_for_model(past_feedback: List[GuessFeedback], system_prompt: str) -> List[dict]:
    """
    Formats the history of guesses into a clean state summary for the model.
    """
    if not past_feedback:
        user_content = "This is the first turn. Please provide your best starting word."
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    known_green = {}
    known_yellow = Counter()
    known_gray = set()

    for fb in past_feedback:
        counts_in_secret_this_turn = Counter()
        for i, f_char in enumerate(fb.feedback.split()):
            if f_char in ('G', 'Y'):
                counts_in_secret_this_turn[fb.guess[i]] += 1
        
        for letter, count in counts_in_secret_this_turn.items():
            known_yellow[letter] = max(known_yellow[letter], count)

        for i, f_char in enumerate(fb.feedback.split()):
            letter = fb.guess[i]
            if f_char == 'G':
                known_green[i] = letter
            elif f_char == 'X':
                if counts_in_secret_this_turn[letter] == 0:
                    known_gray.add(letter)

    green_letters = set(known_green.values())
    for letter in green_letters:
        if letter in known_yellow:
            del known_yellow[letter]
        if letter in known_gray:
            known_gray.remove(letter)
    prompt_parts = ["**Current Knowledge:**"]
    
    # Format Green letters
    green_display = ['_'] * 5
    for idx, letter in known_green.items():
        green_display[idx] = letter
    prompt_parts.append(f"*   **Green Letters (Correct Position):** `{' '.join(green_display)}`")

    # Format Yellow letters
    if known_yellow:
        yellow_display = [f"'{k}' (at least {v})" for k, v in sorted(known_yellow.items())]
        prompt_parts.append(f"*   **Yellow Letters (In word, wrong position):** {', '.join(yellow_display)}")
    else:
        prompt_parts.append(f"*   **Yellow Letters (In word, wrong position):** None")
        
    # Format Gray letters
    if known_gray:
        gray_display = sorted(list(known_gray))
        prompt_parts.append(f"*   **Gray Letters (Not in word):** {', '.join(gray_display)}")
    else:
        prompt_parts.append(f"*   **Gray Letters (Not in word):** None")

    prompt_parts.append("\nBased on this summary, what is your next guess?")
    
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
    Returns:
        A tuple of (game_score, training_reward). The game_score is based solely on the guess quality, 
            the training_reward includes time and length penalties for the loss function.
    """
    reward_config = config.reward
    guess = parse_guess(response)

    # --- Behavioral Penalties ---
    time_penalty = -reward_config.get("time_penalty_per_guess", 1.0)
    length_penalty = 0.0
    if tokenizer and response:
        length_penalty = -(len(tokenizer.encode(response)) * reward_config.get("length_penalty_per_token", 0.01))

    # --- PRIORITY 1 & 2: Handle Game Rule Violations (immediate return) ---
    if not guess:
        game_score = -reward_config.get("format_fail_penalty", 120.0)
        return game_score, game_score + time_penalty + length_penalty

    if guess == secret_word.upper():
        game_score = reward_config.get("solution_correct_guess", 150.0)
        return game_score, game_score + time_penalty + length_penalty
        
    if any(fb.guess == guess for fb in past_feedback):
        game_score = -reward_config.get("repetition_penalty", 30.0)
        return game_score, game_score + time_penalty + length_penalty

    # --- PRIORITY 3: Strategic Score Calculation ---
    # 1. Build a correct, consistent clue state from history.
    known_green = {}  # {index: letter}
    # Use a Counter for yellow letters to handle duplicates correctly.
    known_yellow = Counter()
    known_gray = set()

    for fb in past_feedback:
        guess_letters = list(fb.guess)
        feedback_chars = fb.feedback.split()
        
        # This counter tracks letters that are green or yellow IN THIS SPECIFIC GUESS.
        # This is crucial for correctly identifying which 'X' letters are truly gray.
        counts_in_secret_this_turn = Counter()

        # First pass: Identify all Green and Yellow letters in this turn to establish counts
        for i in range(5):
            letter = guess_letters[i]
            if feedback_chars[i] in ('G', 'Y'):
                counts_in_secret_this_turn[letter] += 1
        
        # Update the global minimum required count for yellow letters
        for letter, count in counts_in_secret_this_turn.items():
            known_yellow[letter] = max(known_yellow[letter], count)

        # Second pass: Process all clues to update the global state
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

    # 2. Crucial Cleanup Step: Finalize the clue state.
    # A letter that is confirmed Green is the highest truth. It cannot also be
    # considered a yellow or gray constraint.
    green_letters = set(known_green.values())
    for letter in green_letters:
        if letter in known_yellow:
            del known_yellow[letter] # No longer a 'yellow' constraint
        if letter in known_gray:
            known_gray.remove(letter) # Cannot be gray if it's green

    # 3. Calculate violations based on the clean, final clue state.
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
    total_penalty += green_violations * reward_config.get("green_position_penalty", 20.0)
    total_penalty += yellow_violations * reward_config.get("yellow_letter_penalty", 15.0)
    total_penalty += gray_violations * reward_config.get("gray_letter_penalty", 15.0)

    # 5. Add soft penalty for out-of-dictionary words.
    if not is_valid_guess(guess, allowed_words):
        total_penalty += reward_config.get("not_in_dictionary_penalty", 25.0)

    # 6. Calculate potential score.
    stagnation_penalty = calculate_stagnation_penalty(guess, known_green, known_yellow, config)
    total_penalty += stagnation_penalty
    strategic_bonus = get_strategic_bonus(guess, past_feedback, config)
    potential_score = reward_config.get("valid_guess_base", 10.0) + strategic_bonus
    
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

        print(f"DEBUG : PLAY_WORDLE: For guess '{best_guess}' against secret '{secret_word}'," 
              f" generated feedback is '{feedback.feedback}'")
        
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
                grouped_by_turn[attempt.prompt_string].append(attempt)
            
            # Ensure we process turns in the order they were played
            turn_prompts = sorted(
                grouped_by_turn.keys(), 
                key=lambda p: game_rollout.attempts.index(grouped_by_turn[p][0])
            )

            # Iterate through each turn and print the winner/loser summary
            for i, prompt in enumerate(turn_prompts):
                attempts_for_turn = grouped_by_turn[prompt]
                if not attempts_for_turn:
                    continue

                # --- FIX: Find the winner that was actually chosen ---
                # The real winner is the one for which we generated feedback to advance the game.
                winner = next((att for att in attempts_for_turn if att.feedback_given is not None), None)

                # If no winner was chosen (e.g., game ended early), fall back to max score
                if winner is None:
                    # This might happen if the game ended because no valid candidates were generated.
                    # In this case, showing the best of the invalid attempts is reasonable.
                    winner = max(attempts_for_turn, key=lambda att: att.game_score)

                # "Losers" are all other generated candidates for this turn.
                losers = [att for att in attempts_for_turn if att is not winner]

                # Format the strings for clean printing
                winner_str = f"'{winner.parsed_guess}' (Score: {winner.game_score:.2f})"
                if losers:
                    losers_str_list = [f"'{l.parsed_guess}' ({l.game_score:.2f})" for l in losers if l.parsed_guess]
                    losers_str = ", ".join(losers_str_list)
                else:
                    losers_str = "None (only one generation)"

                print(f"  - Turn {i+1}: Winner = {winner_str} | Losers = [{losers_str}]")
    return game_rollout