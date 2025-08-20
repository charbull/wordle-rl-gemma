from utils import prompt
from utils import config as cfg
from utils import rl_trainer

if __name__ == "__main__":
    CONFIG_FILE_PATH = "./config/grpo_lora_config.json"
    try:
        config = cfg.load_config_from_file(CONFIG_FILE_PATH)
        print(f"Successfully loaded configuration from {CONFIG_FILE_PATH}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure a valid configuration json file.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the config file: {e}")
        exit()
    rl_trainer.train(config=config, system_prompt=prompt.SYSTEM_PROMPT)