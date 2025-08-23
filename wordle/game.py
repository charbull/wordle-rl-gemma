from dataclasses import dataclass, field
from typing import List
from utils.constants import ALLOWED_GUESSES, FIVE_LETTER_WORD_RE, GUESS_TAG_RE
import re
from collections import Counter
from mlx_lm import generate
from utils import config as cfg
from typing import Callable, Optional
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

# ==============================================================================
# ---  GAME LOGIC FUNCTIONS ---
# ==============================================================================
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
    Formats the history of guesses into a clean state summary, including a list
    of used words and a very explicit final instruction.
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

    prompt_parts = ["You are playing a game of Wordle. Analyze the clues and provide your next guess.",
                    "**Current Knowledge:**"]
    
    green_display = ['_'] * 5
    for idx, letter in known_green.items():
        green_display[idx] = letter
    prompt_parts.append(f"*   **Correct Position (Green):** `{' '.join(green_display)}`")

    if known_yellow:
        yellow_display = [f"'{k}' (at least {v})" for k, v in sorted(known_yellow.items())]
        prompt_parts.append(f"*   **Wrong Position (Yellow):** {', '.join(yellow_display)}")
    else:
        prompt_parts.append(f"*   **Wrong Position (Yellow):** None")
        
    if known_gray:
        gray_display = sorted(list(known_gray))
        prompt_parts.append(f"*   **Not in Word (Gray):** {', '.join(gray_display)}")
    else:
        prompt_parts.append(f"*   **Not in Word (Gray):** None")

    past_guesses = [fb.guess for fb in past_feedback]
    prompt_parts.append(f"*   **Words Already Guessed:** {', '.join(past_guesses)}")
    
    prompt_parts.append("\nYour task is to find a valid 5-letter English word that fits all the clues above.")
    prompt_parts.append("Provide your reasoning within <think> tags, and then your final guess within <guess> tags.")
    
    user_content = "\n".join(prompt_parts)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

def parse_history_from_prompt(user_prompt: str, secret_word: str) -> List[GuessFeedback]:
    """
    Parses a 'Current Knowledge' prompt to reconstruct the game's past feedback.
    """
    # Regex to find the "Words Already Guessed" line and capture the words
    match = re.search(r"\*   \*\*Words Already Guessed:\*\* (.+)", user_prompt)
    if not match:
        return []

    guessed_words_str = match.group(1)
    if guessed_words_str.lower() == 'none':
        return []

    # Split the string of words and strip whitespace
    guessed_words = [word.strip() for word in guessed_words_str.split(',')]

    # Re-calculate the feedback for each past guess to create the history
    past_feedback = []
    for guess in guessed_words:
        if guess: # Ensure we don't process empty strings
            # Use the existing get_feedback function from this file
            feedback_obj = get_feedback(guess, secret_word)
            past_feedback.append(feedback_obj)
    
    return past_feedback

def play_wordle_game(
    model,
    tokenizer,
    secret_word: str,
    system_prompt: str,
    config: cfg.TrainerConfig,
    sampler,
    initial_history: str,
    print_debug = False,
    reward_fn: Optional[Callable] = None,
) -> GameRollout:
    """
    Plays a game of Wordle. If a `reward_fn` is provided, it calculates
    rewards for each generation (for training). Otherwise, it skips reward
    calculation for fast evaluation.

    Args:
        model: The language model to use for generating guesses.
        tokenizer: The tokenizer corresponding to the model.
        secret_word: The secret word to be guessed in this game.
        system_prompt: The system prompt to guide the model's behavior.
        config: Configuration parameters for the RL training and game settings.
        sampler: The sampling strategy to use for generation (e.g., temperature, top-k).
        initial_history: Optional initial history of past guesses and feedback to continue from.
        print_debug: If True, prints detailed debug information during the game.
        reward_fn: Optional custom reward function to be called during training.

    Returns:
        A GameRollout object containing the full history of attempts and whether the game was solved.
    """
    game_rollout = GameRollout(secret_word=secret_word)
    past_feedback: List[GuessFeedback] = []
    already_guessed_words = set()
    max_trials = config.rl.max_trials
    num_generations = config.rl.num_generations

    # This is provided by the dataset for continuing games.
    past_feedback = parse_history_from_prompt(initial_history, secret_word)
    already_guessed_words = {fb.guess for fb in past_feedback}

    if print_debug:
        print(f"\n{'='*35}\n|| NEW GAME || Secret Word: {secret_word.upper()}\n{'='*35}")
        if past_feedback:
            print(f"--- Starting from a history of {len(past_feedback)} turn(s) ---")

    # Game starts here
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
        # TODO: Consider batching these generations when MLX supports it.
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

            game_score, training_reward = 0.0, 0.0
            if print_debug:
                    print(f"  [Generation {i+1}/{num_generations}]")
                    print(f"    Raw Response: \"{response_only}\"")
                    print(f"    Parsed Guess: '{guess}'")
            if reward_fn:
                # The reward function checks for the win condition first.
                game_score, training_reward = reward_fn(
                    response_only, secret_word, past_feedback, config, ALLOWED_GUESSES, tokenizer
                )
                if print_debug:
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
            
        # Filter for only valid, attempts to advance the game state.
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

        # Select the best candidate from the valid ones to advance the game state
        if reward_fn:
            best_attempt = max(valid_candidates, key=lambda att: att.game_score)
        else:
            # If no reward function is provided (evaluation mode), just pick the first valid guess.
            best_attempt = valid_candidates[0]
        best_guess = best_attempt.parsed_guess

        # Generate and append feedback for the chosen path
        is_valid_in_dict = is_valid_guess(best_guess, ALLOWED_GUESSES)
        feedback = get_feedback(best_guess, secret_word)
        feedback.is_in_dictionary = is_valid_in_dict

        print(f"\nwordle play: guess '{best_guess}' against secret '{secret_word}'," 
              f" generated feedback is '{feedback.feedback}'")
        
        if print_debug and reward_fn:
            print("\nTurn Decision:")
            print(f"  - Chose '{best_guess}' (Game Score: {best_attempt.game_score:.2f}) to advance the game.")
            print(f"  - Feedback for next turn: {feedback.feedback}")

        # Add the chosen guess to the set of used words for the next turn's check
        already_guessed_words.add(best_guess)

        # Store the results
        best_attempt.feedback_given = feedback
        # Add all attempts (including failed ones) to the rollout for the trainer.
        game_rollout.attempts.extend(current_turn_attempts)
        # Add only the feedback from the best valid path to guide the next turn.
        past_feedback.append(feedback)

        # 6. Check for game termination
        if best_guess == secret_word.upper():
            print(f"🎉🎉🎉 SUCCESS on attempt {attempt_num + 1}! Word: '{secret_word.upper()}' 🎉🎉🎉")
            game_rollout.solved = True
            break
            
    if not game_rollout.solved:
        if print_debug and reward_fn:
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


def play_eval_game(
    model,
    tokenizer,
    secret_word: str,
    system_prompt: str,
    config: cfg.TrainerConfig,
    sampler,
    model_name: str,
    current_step: int,
    print_debug = False,
) -> GameRecord:
    """
    Calls the game player and correctly processes the resulting
    GameRollout to create an accurate GameRecord for logging.

    Args:
        model: The language model to use for generating guesses.
        tokenizer: The tokenizer corresponding to the model.
        secret_word: The secret word to be guessed in this game.
        system_prompt: The system prompt to guide the model's behavior.
        config: Configuration parameters for the RL training and game settings.
        sampler: The sampling strategy to use for generation (e.g., temperature, top-k).
        model_name: A string identifier for the model (e.g., "Base Model", "LoRA Model").
        current_step: The current training step, for logging purposes.
        print_debug: If True, prints detailed debug information during the game.    
    """
    # For evaluation, we always start with a blank history. The prompt
    # for turn 1 is generated from an empty list of past feedback.
    initial_user_prompt = format_prompt_for_model([], system_prompt)[1]['content']
    
    # Call the powerful, master game-playing function from the trainer
    game_rollout = play_wordle_game(
        model=model,
        tokenizer=tokenizer,
        secret_word=secret_word,
        system_prompt=system_prompt,
        config=config,
        sampler=sampler,
        initial_history=initial_user_prompt,
        print_debug=print_debug,
        # Pass the reward function for the training path, but for eval,
        # we can create a lean version of play_wordle_game that does not need it
        # For now, let's assume we use the universal one with reward_fn=None
        reward_fn=None
    )
    
    if game_rollout.solved:
        turns = len(game_rollout.attempts)
    else:
        turns = config.rl.max_trials # If not solved, it used all turns
    
    # Convert the detailed GameRollout into a simple GameRecord
    return GameRecord(
        log_type=model_name,
        step=current_step,
        secret_word=game_rollout.secret_word,
        solved=game_rollout.solved,
        turns_to_solve=turns, # Use the correctly calculated number of turns
        final_reward=-1.0,
        loss_at_step=-1.0
    )