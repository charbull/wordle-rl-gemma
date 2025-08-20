import utils.config as cfg
import utils.lora as lora
import utils.prompt as prompt
from mlx_lm import generate, load


if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "./config/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "./adapters/grpo_lora_wordle/grpo_lora_wordle_300_20250817-091303.npz"
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