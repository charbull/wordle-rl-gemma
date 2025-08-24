from src.wordle import prompt
from src.utils import config as cfg
from src.ml import rl_trainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GRPO Reinforcement Learning trainer for Wordle.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file for the training run."
    )
    args = parser.parse_args()
    print(f"Using configuration file: {args.config}")
    try:
        config = cfg.load_config_from_file(args.config)
        print(f"Successfully loaded configuration from {args.config}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure a valid configuration json file.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the config file: {e}")
        exit()
    rl_trainer.train(config=config, system_prompt=prompt.SYSTEM_PROMPT)