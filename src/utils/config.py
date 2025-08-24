from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, Optional

@dataclass
class ModelConfig:
    name: str

@dataclass
class TrainingConfig:
    iterations: int
    learning_rate: float
    # Set to True to enable the scheduler
    use_lr_scheduler: bool   
    # The learning rate at the very end of training
    lr_min: float
    # Should match total `iterations`
    lr_decay_steps: int  
    batch_size: int
    log_steps: int
    checkpoint_steps: int
    resume_from_checkpoint: str
    # Path to save the configuration file
    config_file: str  
    # data path for training and evaluation
    data_path: str
    # early stopping
    early_stopping_patience: Optional[int] = None
    use_early_stopping: Optional[bool] = False


@dataclass
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float
    layers_to_tune: int

@dataclass
class RLConfig:
    # we need at least 2 generations to calculate the reward
    num_generations: int = 2
    max_completion_length: int = 256
    sampling_temperature: float = 0.7
    max_trials: int = 6

    def __post_init__(self):
        assert self.num_generations > 1, "num_generations must be greater than 1"


@dataclass
class GRPOConfig:
    clip_epsilon: float = 0.2
    ref_update_steps: int = 20
    kl_coeff: float = 0.02
    beta: float = 0.1
    
@dataclass
class EvalConfig:
    steps: int
    samples: int

@dataclass
class TrainerConfig:
    model: ModelConfig
    training: TrainingConfig
    lora: LoRAConfig
    evaluation: EvalConfig
    # Optional fields with a default value of None
    rl: Optional[RLConfig] = None
    grpo: Optional[GRPOConfig] = None
    reward: Optional[Dict[str, float]] = None

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TrainerConfig":
        rl = None
        grpo = None
        reward = None
        if "rl" in config_dict.keys():
            rl=RLConfig(**config_dict["rl"])
        if "grpo" in config_dict.keys():
            grpo=GRPOConfig(**config_dict["grpo"])
        if "reward" in config_dict.keys():
            reward = config_dict.get("reward", None)

        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            lora=LoRAConfig(**config_dict["lora"]),
            evaluation=EvalConfig(**config_dict["evaluation"]),
            grpo=grpo,
            rl=rl,
            reward=reward
        )

    def to_dict(self) -> Dict:
        return asdict(self)

def save_config(config: TrainerConfig, file_path: Path):
    """Saves the configuration object to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
    print(f"Configuration saved to {file_path}")


def load_config_from_file(config_path: str) -> TrainerConfig:
    """Loads the training configuration from a JSON file."""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
        
    return TrainerConfig.from_dict(config_dict)