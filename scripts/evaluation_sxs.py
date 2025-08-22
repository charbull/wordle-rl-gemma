from utils.rewards_wordle import format_prompt_for_model, get_feedback
from wordle.game import GuessFeedback
from typing import List
import utils.config as cfg
import utils.lora as lora
from utils import read_files
import utils.prompt as prompt
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from utils import constants

def play_wordle_game(
    model,
    tokenizer,
    secret_word: str,
    system_prompt: str,
    max_trials: int = 6,
    print_debug: bool = False
):
    """
    Plays a game of Wordle using the provided model. Injects feedback for invalid
    guesses into the prompt without altering the main feedback history.
    """
    # This list will only store feedback from valid 5-letter guesses.
    past_guesses: List[GuessFeedback] = []
    attempt_num = 0
    already_guessed_words = set()
    
    last_error_message = None

    while attempt_num < max_trials:
       
        # 1. Generate the prompt using ONLY the history of valid guesses.
        messages = format_prompt_for_model(past_guesses, system_prompt)
        
        # 2. If there was an error on the last turn, inject the message now.
        if last_error_message:
            messages[-1]['content'] += f"\n\n{last_error_message}"
            last_error_message = None # Clear the message after using it once.

        if print_debug:
            print(f"💬 sent to model:\n{messages[-1]['content']}")
        prompt_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True
        )
        
        response = generate(model, tokenizer, prompt=prompt_string, max_tokens=2048)
        if print_debug:
            print(f"🤖 Model raw response: {response.strip()}\n")

        match = re.search(r"<guess>(.*?)</guess>", response, re.DOTALL | re.IGNORECASE)
        
        if not match:
            print("⚠️ Model did not use the <guess> tag. This counts as a failed attempt.")
            # Set the error message for the NEXT turn.
            last_error_message = "**Previous Attempt Error:** Your response did not include a guess inside <guess>...</guess> tags. You must provide a guess in the correct format."
            attempt_num += 1
            continue

        guess = re.sub(r'[^A-Z]', '', match.group(1).upper())

        if len(guess) != 5:
            print(f"⚠️ Model's guess '{guess}' is not 5 letters long. This counts as a failed attempt.")
            already_guessed_words.add(guess)
            # Set the error message for the NEXT turn.
            last_error_message = f"**Previous Attempt Error:** Your guess '{guess}' was not 5 letters long. You must guess a 5-letter word."
            attempt_num += 1
            continue

        if guess in already_guessed_words:
            print(f"⚠️ Model's guess '{guess}' is a repeat. This counts as a failed attempt.")
            # Set the error message for the NEXT turn.
            last_error_message = f"**Previous Attempt Error:** You have already guessed the word '{guess}'. You must try a different word."
            attempt_num += 1
            continue
        
        # If we reach here, the guess is a new, valid, 5-letter word.
        already_guessed_words.add(guess)
        feedback = get_feedback(guess, secret_word)
        print(f"🤖 Turn {attempt_num + 1}/{max_trials} Model's guess: '{guess}', feedback: {feedback.feedback}\n")
        past_guesses.append(feedback) # Add the VALID guess to the history.
        attempt_num += 1

        if guess == secret_word.upper():
            print(f"🎉 SUCCESS! The model guessed the secret word {secret_word.upper()} correctly in {attempt_num} attempts! 🎉")
            return {"solved": True, "turns": attempt_num, "secret_word": secret_word}
    
        print("-" * 50)
       
    print(f"❌ FAILURE! The model did not guess the word '{secret_word.upper()}' within {max_trials} trials. ❌")
    return {"solved": False, "turns": max_trials, "secret_word": secret_word}


def plot_comparison_chart(results_df: pd.DataFrame, output_dir: Path):
    """Generates and saves a bar chart comparing model win rates."""
    # Since we know both models played the same number of games, we can
    # simply count the number of unique secret words to find this value.
    if not results_df.empty:
        num_games = results_df['secret_word'].nunique()
    else:
        num_games = 0
    
    plot_title = (
        'Model Performance Comparison: Wordle Win Rate\n'
        f'(Based on {num_games} games per model)'
    )

    # Calculate summary stats
    summary = results_df.groupby('model_name')['solved'].value_counts(normalize=True).unstack(fill_value=0)
    summary['win_rate'] = summary.get(True, 0) * 100
    
    model_names = summary.index
    win_rates = summary['win_rate']
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bars = ax.bar(model_names, win_rates, color=['skyblue', 'orangered'])
    
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    
    # Save the plot
    plot_filename = output_dir / f"model_comparison_wins_num_games_{num_games}.png"
    plt.savefig(plot_filename)
    print(f"\n📈 Comparison plot saved to '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/adapters/grpo_lora_wordle_final_20250820-150425.npz"

    NUM_SAMPLES = 100
    OUTPUT_DIR = Path(LORA_CONFIG_FILE_PATH).parent / "plots"

    # --- 1. Setup: Load words and models ---
    all_possible_words = list(constants.ALLOWED_GUESSES)
    
    random.seed(42)
    secret_words_sample = random.sample(all_possible_words, NUM_SAMPLES)
    
    print(f"Selected a random sample of {NUM_SAMPLES} words for side-by-side evaluation.")


    training_config = cfg.load_config_from_file(LORA_CONFIG_FILE_PATH)
    
    base_model, tokenizer = load(training_config.model.name)
    sampler = make_sampler(temp=training_config.rl.sampling_temperature)
    lora_with_base = lora.load_adapter_with_model(training_config=training_config, adapter_path=LORA_ADAPTER_PATH)
    
    # --- 2. Run Side-by-Side Evaluation ---
    all_results = []
    total_words = len(secret_words_sample)

    win_counts = {'Base Model': 0, 'LoRA Model': 0}

    # Get a handle on the tqdm object to update it dynamically.

    game_progress_bar = tqdm(secret_words_sample, desc="Playing Wordle Games")
    for i, secret_word in enumerate(game_progress_bar):
        print(f"  -> Playing game: (Secret: {secret_word.upper()})")
        
        print_debug = (i % 10 == 0)
        # Play with Base Model
        print(f"  -> Base model playing...")
        base_result = play_wordle_game(base_model, tokenizer, secret_word, prompt.SYSTEM_PROMPT, 6, print_debug)
        base_result['model_name'] = 'Base Model'
        all_results.append(base_result)
        if base_result['solved']:
            win_counts['Base Model'] += 1
        
        print(f"  -> LoRA model playing...")
        # Play with LoRA Model
        lora_result = play_wordle_game(lora_with_base, tokenizer, secret_word, prompt.SYSTEM_PROMPT, 6, print_debug)
        lora_result['model_name'] = 'LoRA Model'
        all_results.append(lora_result)

        if lora_result['solved']:
            win_counts['LoRA Model'] += 1

        postfix_str = f"Wins -> Base: {win_counts['Base Model']}, LoRA: {win_counts['LoRA Model']}"
        game_progress_bar.set_postfix_str(postfix_str)

    # --- 3. Process and Print Results ---
    results_df = pd.DataFrame(all_results)
    
    # Save the raw results to a CSV file for detailed analysis
    csv_path = OUTPUT_DIR / "side_by_side_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📊 Detailed results saved to '{csv_path}'")
    
    # Calculate and print summary statistics
    summary = results_df.groupby('model_name').agg(
        total_wins=('solved', lambda x: x.sum()),
        total_games=('solved', 'count'),
        avg_turns_on_win=('turns', lambda x: x[results_df.loc[x.index, 'solved']].mean())
    ).reset_index()
    summary['win_rate'] = (summary['total_wins'] / summary['total_games']) * 100

    print("\n" + "="*60)
    print(" " * 18 + "SIDE-BY-SIDE EVALUATION RESULTS")
    print("="*60)
    for _, row in summary.iterrows():
        print(f"\n--- {row['model_name']} ---")
        print(f"  Win Rate: {row['win_rate']:.2f}% ({row['total_wins']}/{row['total_games']})")
        print(f"  Avg. Turns on Win: {row['avg_turns_on_win']:.2f}")
    print("\n" + "="*60)
    
    # --- 4. Generate the Comparison Plot ---
    plot_comparison_chart(results_df, OUTPUT_DIR)