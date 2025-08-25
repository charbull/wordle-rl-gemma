import time
from src.wordle import game
from pathlib import Path





if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/adapters/grpo_lora_wordle_final_20250820-150425.npz"

    NUM_SAMPLES = 100
    LOG_INTERVAL = 10
    OUTPUT_DIR = Path(LORA_CONFIG_FILE_PATH).parent / "plots"
    eval_timestamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = OUTPUT_DIR / f"side_by_side_results_{eval_timestamp}.jsonl"
    
    game.play_side_by_side(training_config_path=LORA_CONFIG_FILE_PATH, lora_adapter_path=LORA_ADAPTER_PATH, output_file=jsonl_path, num_samples=NUM_SAMPLES, log_interval=LOG_INTERVAL)