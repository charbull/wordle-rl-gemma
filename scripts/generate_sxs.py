import src.utils.config as cfg
import src.ml.lora as lora
import src.wordle.prompt as prompt
from mlx_lm import generate, load


if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/adapters/grpo_lora_wordle_final_20250820-150425.npz"
    training_config = cfg.load_config_from_file(LORA_CONFIG_FILE_PATH)
    
    
    wordle_messages =[
            {"role" : "system", "content" : prompt.SYSTEM_PROMPT},
           {"role" : "user", "content" : "This is the first turn. Please provide your best starting word."}         
        ]
    
    base_model, tokenizer = load(training_config.model.name)
    prompt = tokenizer.apply_chat_template(
    wordle_messages,
    tokenize=False,
    add_generation_prompt=True,
)
    print("\n--- Testing the Base Model ---")
    base_tokenizer = tokenizer
    base_response = generate(base_model, base_tokenizer, prompt=prompt, max_tokens=2048)
    print("====> base_response: ", base_response)

    print("\n--- Testing the LoRA adapter ---")
    
    merged_model = lora.load_adapter_with_model(training_config=training_config, adapter_path=LORA_ADAPTER_PATH)
    lora_response = generate(merged_model, tokenizer, prompt=prompt, max_tokens=2048)
    print("====> lora_response: ", lora_response)