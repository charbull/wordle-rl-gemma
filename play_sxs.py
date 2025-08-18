from utils.rewards_wordle import play_wordle_game

import utils.config as cfg
import utils.lora as lora
import utils.prompt as prompt
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "./config/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "./adapters/grpo_lora_wordle/grpo_lora_wordle_300_20250817-091303.npz"
    SECRET_WORD = "TABLE"

    training_config = cfg.load_config_from_file(LORA_CONFIG_FILE_PATH)
    
    
    base_model, tokenizer = load(training_config.model.name)
    sampler = make_sampler(temp=training_config.rl.sampling_temperature)
    print("\n--- Testing the Base Model ---")
    game_rollout_base = play_wordle_game(model=base_model,
                                     tokenizer=tokenizer, 
                                     secret_word=SECRET_WORD, 
                                     system_prompt=prompt.SYSTEM_PROMPT, 
                                     config=training_config,
                                     sampler=sampler,
                                     initial_history=None,
                                     print_debug=False)

    print("\n--- Testing the LoRA adapter ---")
    lora_with_base = lora.load_adapter_with_model(training_config=training_config, adapter_path=LORA_ADAPTER_PATH)
    game_rollout_lora = play_wordle_game(model=lora_with_base,
                                     tokenizer=tokenizer, 
                                     secret_word=SECRET_WORD, 
                                     system_prompt=prompt.SYSTEM_PROMPT, 
                                     config=training_config,
                                     sampler=sampler,
                                     initial_history=None,
                                     print_debug=False)