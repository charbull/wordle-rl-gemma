import json
import time
from wordle import game
from typing import List
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import constants
from utils.logging import plot_comparison_chart, write_metrics_to_file
import utils.config as cfg
import ml.lora as lora
import wordle.prompt as prompt



if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/adapters/grpo_lora_wordle_final_20250820-150425.npz"

    NUM_SAMPLES = 2
    LOG_INTERVAL = 10
    OUTPUT_DIR = Path(LORA_CONFIG_FILE_PATH).parent / "plots"
    eval_timestamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = OUTPUT_DIR / f"side_by_side_results_{eval_timestamp}.jsonl"
    
    results_buffer: List[game.GameRecord] = []
    win_counts = {'Base Model': 0, 'LoRA Model': 0}
    
    # --- 1. Setup: Load words and models ---
    all_possible_words = list(constants.ANSWERS_WORDS)
    
    random.seed(42)
    secret_words_sample = random.sample(all_possible_words, NUM_SAMPLES)
    
    print(f"Selected a random sample of {NUM_SAMPLES} words for side-by-side evaluation.")


    training_config = cfg.load_config_from_file(LORA_CONFIG_FILE_PATH)
    training_config.rl.sampling_temperature = 0.0  # Deterministic sampling for evaluation
    training_config.rl.num_generations = 1  # Single generation per prompt

    # --- 2. Prepare the LoRA Model ---
    print("\nLoading and preparing the LoRA-finetuned model...")
    lora_foundation_model, tokenizer = load(training_config.model.name)
    lora_config = {"rank": training_config.lora.rank, "alpha": training_config.lora.alpha, "dropout": training_config.lora.dropout}
    
    lora_model_with_layers = lora.apply_lora_to_model(
        model=lora_foundation_model, 
        lora_config=lora_config, 
        layers_to_tune=training_config.lora.layers_to_tune
    )
    lora_model = lora.load_adapter(model=lora_model_with_layers, adapter_path=LORA_ADAPTER_PATH)
    
    # --- 3. Load a SEPARATE, CLEAN Base Model for Comparison ---
    print("\nLoading a clean base model for comparison...")
    base_model, _ = load(training_config.model.name)
    
    # --- 4. Run the Verification Check ---
    lora.verify_lora_loading(base_model, lora_model)
    
    # --- 5. Run Side-by-Side Evaluation (as before) ---
    print("\nStarting side-by-side evaluation...")
    # Create the deterministic sampler for evaluation
    eval_sampler = make_sampler(temp=0.0)
    
    # --- 2. Run Side-by-Side Evaluation ---
    total_words = len(secret_words_sample)


    # Get a handle on the tqdm object to update it dynamically.
    pbar = tqdm(enumerate(secret_words_sample), total=NUM_SAMPLES, desc="Playing Wordle Games")
    for i, secret_word in pbar:
        print(f"  -> Playing game: (Secret: {secret_word.upper()})")
        
        print_debug = (i % 10 == 0)
        print(f"  -> Base model playing...")
        base_record = game.play_eval_game(base_model, tokenizer, secret_word, prompt.SYSTEM_PROMPT, training_config, eval_sampler, 'Base Model', i, print_debug)
        results_buffer.append(base_record)
        if base_record.solved:
            win_counts['Base Model'] += 1
        
        print(f"  -> LoRA model playing...")
        lora_record = game.play_eval_game(lora_model, tokenizer, secret_word, prompt.SYSTEM_PROMPT, training_config, eval_sampler, 'LoRA Model', i, print_debug)      
        results_buffer.append(lora_record)
        if lora_record.solved:
            win_counts['LoRA Model'] += 1

        pbar.set_postfix_str(f"Wins -> Base: {win_counts['Base Model']}, LoRA: {win_counts['LoRA Model']}")

        if (i + 1) % LOG_INTERVAL == 0:
            write_metrics_to_file(results_buffer, jsonl_path)
            results_buffer.clear()

    if results_buffer:
        write_metrics_to_file(results_buffer, jsonl_path)
        results_buffer.clear()
        
    print(f"\n📊 Detailed results saved to '{jsonl_path}'")

    # --- Process and Print Final Results ---
    with open(jsonl_path, 'r') as f:
        all_results = [json.loads(line) for line in f]
    results_df = pd.DataFrame(all_results)
    
    # Use 'log_type' to group data by model
    summary = results_df.groupby('log_type').agg(
        total_wins=('solved', lambda x: x.sum()),
        total_games=('solved', 'count'),
        avg_turns_on_win=('turns_to_solve', lambda x: x[results_df.loc[x.index, 'solved']].mean())
    ).reset_index()
    summary['win_rate'] = (summary['total_wins'] / summary['total_games']) * 100
    
    print("\n" + "="*60 + "\n" + " " * 18 + "SIDE-BY-SIDE EVALUATION RESULTS" + "\n" + "="*60)
    for _, row in summary.iterrows():
        print(f"\n--- {row['log_type']} ---") # Use log_type for model name
        print(f"  Win Rate: {row['win_rate']:.2f}% ({row['total_wins']}/{row['total_games']})")
        print(f"  Avg. Turns on Win: {row['avg_turns_on_win']:.2f}")
    print("\n" + "="*60)
    
    plot_comparison_chart(results_df, OUTPUT_DIR, timestamp=eval_timestamp)