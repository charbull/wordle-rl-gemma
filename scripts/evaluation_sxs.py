import time
from src.wordle import game
from typing import List
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from pathlib import Path
from tqdm import tqdm
from src.utils.logging import plot_comparison_chart, write_metrics_to_file, plot_cumulative_wins_sxs, plot_win_distribution_sxs
import src.utils.config as cfg
import src.ml.lora as lora
import src.wordle.prompt as prompt

def play_side_by_side(training_config_path: str, lora_adapter_path, metrics_file: str, num_samples: int = 100, log_interval: int = 10, with_game_history: bool = True):
    """ Play side by side LoRA vs Base model on a num_samples"""
    results_buffer: List[game.GameRecord] = []
    win_counts = {'Base Model': 0, 'LoRA Model': 0}
    
    training_config = cfg.load_config_from_file(training_config_path)
    training_config.rl.sampling_temperature = 0.0  # Deterministic sampling for evaluation
    training_config.rl.num_generations = 1  # Single generation per prompt
    # get the same split used in training and use the test set to make sure there is no data contamination
    _, _, test_dataset = game.prepare_data(config=training_config)

    print(f"Selected a random sample of {num_samples} words for side-by-side evaluation from the test dataset of {len(test_dataset)} samples.")

    # --- 2. Prepare the LoRA Model ---
    print("\nLoading and preparing the LoRA-finetuned model...")
    lora_foundation_model, tokenizer = load(training_config.model.name)    
    lora_model_with_layers = lora.apply_lora_to_model(
        model=lora_foundation_model, 
        lora_config=training_config.lora
    )
    lora_model = lora.load_adapter(model=lora_model_with_layers, adapter_path=lora_adapter_path)
    
    # --- 3. Load a Base Model for Comparison ---
    print("\nLoading a clean base model for comparison...")
    base_model, _ = load(training_config.model.name)

    # --- 4. Run the Verification Check ---
    # lora.verify_lora_loading(base_model, lora_model)
    
    # --- 5. Run Side-by-Side Evaluation ---
    print("\nStarting side-by-side evaluation...")
    # Create the deterministic sampler for evaluation
    eval_sampler = make_sampler(temp=0.1)
    
    # Use the .select() method to create a view of the first `num_samples` rows.
    # This is very fast and does not load all the data into RAM.
    subset_dataset = test_dataset.select(range(num_samples))

    # --- 2. Run Side-by-Side Evaluation ---
    # Get a handle on the tqdm object to update it dynamically.
    pbar = tqdm(enumerate(subset_dataset), total=num_samples, desc="Playing Wordle Games")
    for i, sample in pbar:
        secret_word = sample['secret']
        print(f"  -> Playing game: (Secret: {secret_word.upper()})")
    
        # Evaluation, we can start blank history. The prompt
        # for turn 1 is generated from an empty list of past feedback.
        if with_game_history:
            initial_history = sample['messages'][1]['content']
        else:
            initial_history = None
        
        print_debug = (i % 10 == 0)
        print(f"  -> Base model playing...")
        base_record = game.play_eval_game(model=base_model, tokenizer=tokenizer, secret_word=secret_word, 
                                     system_prompt=prompt.SYSTEM_PROMPT, config=training_config,
                                    sampler=eval_sampler, model_name='Base Model', current_step=i, print_debug=print_debug, 
                                    initial_history=initial_history)
        results_buffer.append(base_record)
        if base_record.solved:
            win_counts['Base Model'] += 1
        
        print(f"  -> LoRA model playing...")
        lora_record = game.play_eval_game(model=lora_model, tokenizer=tokenizer, secret_word=secret_word, 
                                     system_prompt=prompt.SYSTEM_PROMPT, config=training_config,
                                    sampler=eval_sampler, model_name='LoRA Model', current_step=i, print_debug=print_debug,
                                    initial_history=initial_history)    
        results_buffer.append(lora_record)
        if lora_record.solved:
            win_counts['LoRA Model'] += 1

        pbar.set_postfix_str(f"Wins -> Base: {win_counts['Base Model']}, LoRA: {win_counts['LoRA Model']}")

        if (i + 1) % log_interval == 0:
            write_metrics_to_file(results_buffer, metrics_file)
            results_buffer.clear()

    if results_buffer:
        write_metrics_to_file(results_buffer, metrics_file)
        results_buffer.clear()
        
    print(f"\n📊 Detailed results saved to '{metrics_file}'")
    plot_comparison_chart(metrics_file)

    plot_cumulative_wins_sxs(metrics_file)
    plot_win_distribution_sxs(metrics_file)



if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250824-133827_gemma-3-4b-it-bf16_rank64/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250824-133827_gemma-3-4b-it-bf16_rank64/adapters/adapter_step_500.npz"

    NUM_SAMPLES = 150
    LOG_INTERVAL = 10
    OUTPUT_DIR = Path(LORA_CONFIG_FILE_PATH).parent / "plots"
    eval_timestamp = time.strftime("%Y%m%d-%H%M%S")

    jsonl_path = OUTPUT_DIR / f"side_by_side_results_{eval_timestamp}_without_history_temp01.jsonl"
    play_side_by_side(training_config_path=LORA_CONFIG_FILE_PATH, lora_adapter_path=LORA_ADAPTER_PATH, metrics_file=jsonl_path, num_samples=NUM_SAMPLES, log_interval=LOG_INTERVAL, with_game_history=False)
