import json
import time
import math
import random
from typing import List, Dict
from src.wordle import rewards
from src.wordle.game import get_feedback, format_prompt_for_model, find_valid_completions, get_clue_summary


# ==============================================================================
# Wordle Game Simulator and Helper Functions
# ==============================================================================

class WordleGame:
    """A class to simulate a game of Wordle."""
    def __init__(self, secret_word: str, valid_guesses: List[str], max_turns: int = 6):
        self.secret_word = secret_word.lower()
        self.valid_guesses = set(valid_guesses)
        self.guesses = []
        self.feedback = []
        self.turn = 1
        self.is_over = False
        self.max_turns = max_turns

    def make_guess(self, guess: str):
        if self.is_over: return
        self.guesses.append(guess)
        self.feedback.append(get_feedback(guess=guess, secret_word=self.secret_word).feedback)
        if guess == self.secret_word.upper() or self.turn >= self.max_turns:
            self.is_over = True
        self.turn += 1





# ==============================================================================
# 3. Prompt and Chain-of-Thought Generation
# ==============================================================================

def format_prompt_cot(w_game: WordleGame, chosen_completion_word: str, prompt: str) -> List[Dict]:
    """
    Formats the history of guesses into a user-facing prompt string, there is no target data points for RL training.
    This is used to generate the Chain-of-Thought (CoT) reasoning block.
    The prompt is structured to include all previous guesses and feedback,
    and the chosen word for the next guess.
    
    Args:
        game (WordleGame): The current game state.
        chosen_completion_word (str): The word that the model should guess next.
        prompt (str): The system prompt to use in the conversation.

    Returns:
        List[Dict]: A list of messages formatted for the model, including the system prompt and user history.
    """
    guess_feedbacks = []
    if chosen_completion_word:
        guess_feedback = get_feedback(guess=chosen_completion_word, secret_word=w_game.secret_word)  # Ensure feedback is generated for the chosen word
        guess_feedbacks.append(guess_feedback)
    for guess, feedback in zip(w_game.guesses, w_game.feedback):
        guess_feedbacks.append(rewards.GuessFeedback(
            guess=guess,
            feedback=feedback
        ))
    format_prompt_msgs = format_prompt_for_model(past_feedback=guess_feedbacks, 
                                                  system_prompt=prompt)
    return format_prompt_msgs


def generate_think_block(clues: Dict, completion: str, num_valid_options: int) -> str:
    """Generates a structured, rule-based reasoning block."""
    think_parts = []

    # Step 1: Analyze Greens
    green_str = "".join(clues['greens']).upper()
    if '_' in green_str:
        think_parts.append(f"From the clues, I know the word has the structure {green_str}.")
    else:
        think_parts.append("All letter positions are known.")

    # Step 2: Analyze Yellows
    if clues['yellows']:
        yellow_str = ", ".join(sorted(list(clues['yellows']))).upper()
        think_parts.append(f"The word must also contain the letter(s) {yellow_str} but not in different positions.")

    # Step 3: Analyze Greys
    if clues['greys']:
        grey_str = ", ".join(sorted(list(clues['greys']))).upper()
        think_parts.append(f"I must avoid all eliminated letters: {grey_str}.")

    # Step 4: State Strategy
    if len(clues['greens']) - green_str.count('_') + len(clues['yellows']) < 2:
         think_parts.append("The board is quite open, so my goal is to test common vowels and consonants to gather more information.")
    elif num_valid_options > 10:
        think_parts.append(f"There are still {num_valid_options} possibilities. I need a strategic guess to narrow down the options effectively.")
    else:
        think_parts.append(f"The options are now very limited (only {num_valid_options} known possibilities). My goal is to pinpoint the exact word.")

    # Step 5: Justify the chosen completion
    think_parts.append(f"After reviewing the options, the word '{completion.upper()}' is an excellent choice as it fits all the known criteria perfectly.")

    full_think_block = " ".join(think_parts)
    return f"<think>{full_think_block}</think>"


def find_best_guess(possible_words: list[str], allowed_guesses: list[str]) -> str:
    """Finds the best guess to maximize information gain (entropy)."""
    if not possible_words:
        return ""
    if len(possible_words) <= 2:
        return possible_words[0]

    best_guess = ""
    max_entropy = -1

    # To speed up, we can limit the pool of words we check as potential guesses
    guess_pool = allowed_guesses
    if len(possible_words) > 100:
        # A common heuristic is to only check from possible answers when the list is big
        guess_pool = list(set(allowed_guesses) & set(possible_words))
        if not guess_pool: guess_pool = allowed_guesses

    for guess in guess_pool:
        groups = {}
        for answer in possible_words:
            guess_feedback = get_feedback(guess=guess, secret_word=answer)
            feedback_tuple = (guess_feedback.guess, guess_feedback.feedback)
            groups.setdefault(feedback_tuple, 0)
            groups[feedback_tuple] += 1

        entropy = 0
        total_possible = len(possible_words)
        for feedback in groups:
            p = groups[feedback] / total_possible
            entropy -= p * math.log2(p)

        if entropy > max_entropy:
            max_entropy = entropy
            best_guess = guess

    return best_guess


# ==============================================================================
# 4. Main Data Generation Script
# ==============================================================================
def generate_cot_sft_data(num_samples: int, output_file: str, prompt: str, solutions_words: List[str]):
    """Main function to generate the Chain-of-Thought SFT dataset. 
    It creates a game state and a CoT data point for SFT training that 
    contains a think block and a guess to be used as a target for the model training.

    Args:
        num_samples (int): The number of samples to generate.
        output_file (str): The file path to save the generated data.
        prompt (str): The system prompt to use in the conversation.
        solutions_words (List[str]): A list of valid solution words for Wordle.
    """
    print(f"Starting CoT data generation for {num_samples} samples...")
    start_time = time.perf_counter()
    with open(output_file, 'w') as f:
        generated_count = 0
        while generated_count < num_samples:
                secret_word = random.choice(solutions_words)
                game = WordleGame(secret_word, solutions_words)

                # simulate a random number of turns to create a game state.
                # The SFT data point will be the *next* logical step from this state.
                num_previous_turns = random.randint(0, 4)
                for i in range(num_previous_turns):
                    if game.is_over: break

                    # Make a reasonable guess to advance the game state. Using a fast
                    # random choice is fine here; we just need to create a plausible history.
                    if i == 0:
                        sim_guess = "SOARE" # Good starter
                    else:
                        current_clues = get_clue_summary(game.guesses, game.feedback)
                        possible_answers_for_sim = find_valid_completions(current_clues, solutions_words)
                        if not possible_answers_for_sim: break
                        sim_guess = random.choice(possible_answers_for_sim)
                    
                    if not sim_guess: break
                    game.make_guess(sim_guess)

                if game.is_over: 
                    continue

                # --- SFT DATA POINT GENERATION ---
                # 1. Get all clues and find the remaining valid words.
                clues = get_clue_summary(game.guesses, game.feedback)
                completions = find_valid_completions(clues, solutions_words)
                if not completions: continue

                # 2. Find the *best* next guess. This is our target for the assistant.
                if not game.guesses:
                    chosen_completion_word = "SOARE"
                else:
                    chosen_completion_word = find_best_guess(completions, solutions_words)
                
                if not chosen_completion_word:
                    chosen_completion_word = completions[0] # Fallback if best guess fails

                # 3. Generate the user-facing prompt (the game history)
                prompt_messages = format_prompt_cot(game, chosen_completion_word, prompt)

                # 4. Generate the CoT "think" block based on the state and the *chosen* word
                think_block = generate_think_block(clues, chosen_completion_word, len(completions))

                # 5. Assemble the final assistant response
                final_completion = f"{think_block}<guess>{chosen_completion_word.strip()}</guess>"

                # 6. Create and write the final JSON object
                messages = [
                   prompt_messages[0],
                    {"role": "assistant", "content": final_completion},
                ]
                data = {'messages': messages, "secret": secret_word.strip()}
                f.write(json.dumps({"data": data}) + "\n")

                generated_count += 1
                if generated_count % 50 == 0:
                    print(f"  Generated {generated_count}/{num_samples} samples...")

    print(f"\nSuccessfully generated SFT {generated_count} samples and saved to '{output_file}'.")
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f' Took {delta_time:.4f} seconds to generate {num_samples} data samples')


def generate_cot_rl_data(num_samples: int, output_file: str, prompt: str, solution_words: List[str]):
    """Main function to generate the Chain-of-Thought SFT dataset."""
    print(f"Starting CoT data generation for {num_samples} samples...")
    start_time = time.perf_counter()
    with open(output_file, 'w') as f:
        generated_count = 0
        while generated_count < num_samples:
                secret_word = random.choice(solution_words)
                game = WordleGame(secret_word, solution_words)

                # simulate a random number of turns to create a game state.
                # The SFT data point will be the *next* logical step from this state.
                num_previous_turns = random.randint(0, 4)
                for i in range(num_previous_turns):
                    if game.is_over: break

                    # Make a reasonable guess to advance the game state. Using a fast
                    # random choice is fine here; we just need to create a plausible history.
                    if i == 0:
                        sim_guess = "SOARE" # Good starter
                    else:
                        current_clues = get_clue_summary(game.guesses, game.feedback)
                        possible_answers_for_sim = find_valid_completions(current_clues, solution_words)
                        if not possible_answers_for_sim: break
                        sim_guess = random.choice(possible_answers_for_sim)
                    
                    if not sim_guess: break
                    game.make_guess(sim_guess)

                if game.is_over: 
                    continue

                # 1. Get all clues and find the remaining valid words.
                clues = get_clue_summary(game.guesses, game.feedback)
                completions = find_valid_completions(clues, solution_words)
                if not completions: continue

                # 2. Find the *best* next guess. This is our target for the assistant.
                if not game.guesses:
                    chosen_completion_word = "SOARE"
                else:
                    chosen_completion_word = find_best_guess(completions, solution_words)
                    if chosen_completion_word == secret_word:
                        continue
                
                if not chosen_completion_word:
                    chosen_completion_word = completions[0] # Fallback if best guess fails

                # 3. Generate the user-facing prompt (the game history)
                messages = format_prompt_cot(game, chosen_completion_word, prompt)

                # 4. Create and write the final JSON object
                data = {'messages': messages, "secret": secret_word.strip()}
                f.write(json.dumps({"data": data}) + "\n")

                generated_count += 1
                if generated_count % 50 == 0:
                    print(f"  Generated {generated_count}/{num_samples} samples...")

    print(f"\nSuccessfully generated RL {generated_count} samples and saved to '{output_file}'.")
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print(f' Took {delta_time:.4f} seconds to generate {num_samples} data samples')
    