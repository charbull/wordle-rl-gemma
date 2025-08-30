# We define the LoRALinear class here to avoid importing it from mlx-examples,
# making the script fully self-contained.
from typing import Any
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import math
import src.utils.config as cfg
import os
from mlx.utils import tree_flatten
from mlx.utils import tree_flatten, tree_unflatten
import json
from mlx_lm import load
import functools




class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            lora_rank=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
        )
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self):
        linear = self.linear
        
        # check for the bias
        has_bias = hasattr(linear, "bias") and linear.bias is not None

        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)
        original_dtype = weight.dtype

        if is_quantized:
            weight = mx.dequantize(
                weight, linear.scales, linear.biases, linear.group_size, linear.bits
            )

        # The LoRA update is W' = W + scale * (B.T @ A.T)
        lora_a_t = self.lora_a.T
        lora_b_t = self.lora_b.T

        # matrix multiplication: (out, rank) @ (rank, in) -> (out, in)
        delta_w = (self.scale * lora_b_t) @ lora_a_t
        print(f"Delta weights shape: {delta_w.shape}, dtype: {delta_w.dtype}")
        fused_weight = weight.astype(delta_w.dtype) + delta_w
        print(f"Fused weights shape: {fused_weight.shape}, dtype: {fused_weight.dtype}")
        
        output_dims, input_dims = fused_weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=has_bias)
        fused_linear.weight = fused_weight.astype(original_dtype)
        if has_bias:
            fused_linear.bias = linear.bias.astype(original_dtype)

        if is_quantized:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear, linear.group_size, linear.bits
            )
        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.lora_rank = lora_rank
        # use rsLoRA scaling factor: scale = alpha / rank
        self.scale = lora_alpha / math.sqrt(lora_rank)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale, high=init_scale, shape=(input_dims, lora_rank)
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))
        
    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        # Apply dropout to the first LoRA matrix multiplication
        z = self.dropout(x @ self.lora_a) @ self.lora_b
        return y + self.scale * z.astype(y.dtype)
    


def save_adapter(model: nn.Module, output_dir: str, lora_config: cfg.LoRAConfig, model_name: str):
    """
    Saves the LoRA adapter weights and the required configuration file to a directory.
    This makes it compatible with `mlx_lm.load(adapter_path=...)`.

    Args:
        model (nn.Module): The model with LoRA layers.
        output_dir (str): The directory to save the adapter files to.
        lora_config (dict): The dictionary containing LoRA parameters (rank, alpha, etc.). this is for the json.
        model_name (str): The name of the base model used for training.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    weights_path = os.path.join(output_dir, "adapter_model.safetensors")
    config_path = os.path.join(output_dir, "adapter_config.json")

    # Pass the string paths to the save/open functions
    model.save_weights(weights_path)

    config_for_saving = {
        "base_model_name_or_path": model_name,
        "peft_type": "LORA",
        "r": lora_config.rank,
        "lora_alpha": lora_config.alpha,
        "lora_dropout": lora_config.dropout,
        "target_modules": lora_config.layers_to_tune,
        "bias": "none",
    }

    with open(config_path, "w") as f:
        json.dump(config_for_saving, f, indent=4)

    print(f"✅ Adapter successfully saved to {output_dir}")

def find_transformer_layers(model: nn.Module) -> list:
    """
    Programmatically finds the list of transformer layers in a model,
    handling different known architectures.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        list: The list of transformer layer modules.
    
    Raises:
        ValueError: If no known layer path is found.
    """
    # List of potential paths to the transformer layers, ordered by preference
    # or specificity.
    potential_paths = [
        "language_model.model.layers",  # For gemma-3-4b
        "model.layers",                 # For gemma-1b, gemma-3-270m, and many others
    ]

    for path in potential_paths:
        try:
            # This uses a functional approach to safely access nested attributes.
            # It's equivalent to model.language_model.model.layers
            layers = functools.reduce(getattr, path.split('.'), model)
            
            # Check if we actually found a non-empty list of modules
            if isinstance(layers, list) and len(layers) > 0 and isinstance(layers[0], nn.Module):
                print(f"✅ Found transformer layers at path: 'model.{path}'")
                return layers
        except AttributeError:
            # This path doesn't exist, so we'll just try the next one.
            continue
            
    # If the loop finishes and we found nothing, raise a helpful error.
    raise ValueError(
        "Could not automatically find transformer layers in the model. "
        "Please inspect the model architecture by printing it, and "
        "add the correct path to the `potential_paths` list in the "
        "`find_transformer_layers` function in lora.py."
    )


def apply_lora_to_model(model: nn.Module, lora_config: cfg.LoRAConfig)-> nn.Module:
    """ Applies LoRA structure to the model's layers.
    Note: the gemman models have different layer access patterns.
    Its important to print the model architecture to confirm the correct layers are being modified.
    Args:
        model (nn.Module): The base model to apply LoRA to.
        lora_config: A config of LoRA parameters: rank, alpha, dropout.
    Returns:
        nn.Module: The model with LoRA layers applied.
    
    """
    # Freeze the model before applying LoRA
    model.freeze()
    # print(model)
    mlx_lora_config = {"rank": lora_config.rank, "alpha": lora_config.alpha, "dropout": lora_config.dropout}

    layers = find_transformer_layers(model)
    min_layers_to_tune = min(lora_config.layers_to_tune, len(layers)) # gemma 4b
    if min_layers_to_tune < lora_config.layers_to_tune:
        print(f"Warning: Trying to tune {lora_config.layers_to_tune} layers, but model only has {len(model.model.layers)}. Tuning {min_layers_to_tune} layers instead.")
    for l in layers[-min_layers_to_tune:]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj, **mlx_lora_config)
        l.self_attn.k_proj = LoRALinear.from_linear(l.self_attn.k_proj, **mlx_lora_config)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj, **mlx_lora_config)
        l.self_attn.o_proj = LoRALinear.from_linear(l.self_attn.o_proj, **mlx_lora_config)

    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(
        f"LoRA trainable parameters: {trainable_params / 1e6:.3f}M | "
        f"Total model parameters: {total_params / 1e6:.3f}M | "
        f"Percentage: {(trainable_params / total_params) * 100:.4f}%"
    )
    return model

def get_named_parameters_flat(model_params: dict, prefix: str = ''):
    """
    A helper function to recursively flatten a nested structure of parameters
    and return a list of (name, value) pairs.
    """
    flat_params = []
    for name, param in model_params.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(param, dict):
            flat_params.extend(get_named_parameters_flat(param, prefix=full_name))
        elif isinstance(param, list):
            for i, sub_param in enumerate(param):
                flat_params.extend(get_named_parameters_flat(sub_param, prefix=f"{full_name}.{i}"))
        else:
            flat_params.append((full_name, param))
    return flat_params


# TODO update to use safetensors
def save_checkpoint(model: nn.Module, checkpoint_file: str):
    """
    Saves the LoRA adapter weights as a .npz file.
    """
    # Get the trainable parameters (which are only the LoRA weights
    #    since the base model is frozen) and flatten them into a dictionary.
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    
    # Save the weights.
    mx.savez(checkpoint_file, **adapter_weights)
    
    print(f"\n--- Checkpoint saved to {checkpoint_file} ---")
    
    return checkpoint_file

def load_adapter(model: nn.Module, adapter_path: str):
    """
    Loads LoRA adapter weights into a model.
    """
    print(f"Loading adapter weights from: {adapter_path}")
    try:
        # Load the dictionary of adapter weights (e.g., {'layers.0.attention.q_proj.lora_a': array(...)})
        adapter_weights = mx.load(adapter_path)
        print(f"Adapter weights loaded successfully. len Keys: {len(adapter_weights.keys())}")
        if not adapter_weights:
            raise ValueError("Warning: Adapter weights are empty.")
    except FileNotFoundError:
        print(f"Error: Adapter file not found at '{adapter_path}'")
        return model

    # The model's parameters are a nested dictionary. We need to update it.
    # The `tree_unflatten` utility is perfect for this. It takes a flat list of
    # (key, value) pairs and reconstructs the nested structure.
    model.update(tree_unflatten(list(adapter_weights.items())))
    
    # Ensure the updates are applied
    mx.eval(model.parameters())
    
    print("Adapter loading complete.")
    return model


def load_adapter_with_model(training_config, adapter_path):
    """Loads a LoRA adapter with the base model.

    Args:
        training_config (cfg.TrainerConfig): The training configuration containing model and LoRA settings.
        adapter_path (str): The path to the LoRA adapter directory.

    Returns:
        nn.Module: The base model with the LoRA adapter applied and loaded.
    """
    print(f"Loading base model: {training_config.model.name}")
    print(f"Loading LoRA adapter from: {adapter_path}")
    
    base_model_name = training_config.model.name
    lora_config = training_config.lora

    base_model, _ = load(base_model_name)
    lora_adapter_with_base_model = apply_lora_to_model(base_model, lora_config)
    lora_adapter_with_base_model = load_adapter(lora_adapter_with_base_model, adapter_path)

    return lora_adapter_with_base_model


def verify_lora_loading(base_model: nn.Module, lora_model: nn.Module) -> bool:
    """
    Verifies that the LoRA adapter has been loaded correctly by performing two checks:
    1.  Confirms a non-adapted layer's weights are identical between the two models.
    2.  Confirms that a LoRA-specific weight matrix in an adapted layer is not all zeros.

    Args:
        base_model (nn.Module): The clean, original model.
        lora_model (nn.Module): The model after applying LoRA and loading an adapter.

    Returns:
        bool: True if the LoRA adapter is loaded correctly, False otherwise.
    """
    print("Verifying LoRA adapter loading...")
    for name, base_param in base_model.named_parameters():
        lora_param = dict(lora_model.named_parameters()).get(name)
        if lora_param is None:
            print(f"Parameter '{name}' not found in LoRA model.")
            continue

        if mx.sum(base_param != lora_param).item() == 0:
            print(f"Parameter '{name}' is identical in both models (not adapted).")
        else:
            print(f"Parameter '{name}' differs between models (adapted).")

    for name, lora_param in lora_model.named_parameters():
        if "lora" in name and mx.sum(mx.abs(lora_param)).item() == 0:
            print(f"Warning: LoRA-specific parameter '{name}' is all zeros.")

    print("LoRA adapter verification complete.")
    return True