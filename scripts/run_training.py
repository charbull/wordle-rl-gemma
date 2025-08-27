import argparse
from src.utils import config as cfg
from src.wordle.prompt import SYSTEM_PROMPT
from src.ml.grpo_trainer import GRPOTrainer
from src.ml.gspo_trainer import GSPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train a Wordle solver using RL.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["grpo", "gspo"],
        required=True,
        help="The reinforcement learning algorithm to use.",
    )
    args = parser.parse_args()

    # Load configuration
    config = cfg.load_config_from_file(args.config)
   
    print(f"Initializing trainer for algorithm: {args.algo.upper()}")
    
    # Select and instantiate the trainer based on the chosen algorithm
    if args.algo == "grpo":
        trainer = GRPOTrainer(config, SYSTEM_PROMPT)
    elif args.algo == "gspo":
        trainer = GSPOTrainer(config, SYSTEM_PROMPT)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()