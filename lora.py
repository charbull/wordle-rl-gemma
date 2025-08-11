# ==============================================================================
# ---  LoRA Code (Copied directly from mlx-examples) ---
# ==============================================================================
# We define the LoRALinear class here to avoid importing it from mlx-examples,
# making the script fully self-contained.
from typing import Any
from functools import reduce
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from pathlib import Path
import math
import src.config as cfg

from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from safetensors.mlx import save_file as save_safetensors
import json
from dataclasses import asdict

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

        # The LoRA update is W' = W + scale * (A @ B).T
        # which is equivalent to W' = W + scale * (B.T @ A.T)
        lora_a_t = self.lora_a.T
        lora_b_t = self.lora_b.T

        # Correct matrix multiplication: (out, rank) @ (rank, in) -> (out, in)
        delta_w = (self.scale * lora_b_t) @ lora_a_t
        fused_weight = weight.astype(delta_w.dtype) + delta_w
        
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
        # The scaling factor is a key parameter in LoRA.
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
        lora_config (dict): The dictionary containing LoRA parameters (rank, alpha, etc.).
        model_name (str): The name of the base model used for training.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save the trainable LoRA weights to a safetensors file
    # This correctly extracts only the parameters that were trained.
    model.save_weights(str(output_path / "adapter_model.safetensors"))

    # 2. Create and save the adapter_config.json file
    config_for_saving = {
        "base_model_name_or_path": model_name,
        "peft_type": "LORA",
        "r": lora_config.rank,
        "lora_alpha": lora_config.alpha,
        "lora_dropout": lora_config.dropout,
        "target_modules": lora_config.layers_to_tune,
        "bias": "none",  # Common default for LoRA
    }

    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(config_for_saving, f, indent=4)

    print(f"✅ Adapter successfully saved to {output_path}")

def apply_lora_to_model(model: nn.Module, lora_config: dict[str, Any], layers_to_tune: int)-> nn.Module:
    """ Applies LoRA structure to the model's layers."""
    # Freeze the model before applying LoRA
    model.freeze()
    # print(model.language_model.model)
    layers = model.language_model.model.layers # gemma 4b
    # layers = model.model.layers # gemma 1b
    # min_layers_to_tune = min(layers_to_tune, len(model.model.layers)) # gemma 1b
    min_layers_to_tune = min(layers_to_tune, len(layers)) # gemma 4b
    if min_layers_to_tune < layers_to_tune:
        print(f"Warning: Trying to tune {layers_to_tune} layers, but model only has {len(model.model.layers)}. Tuning {min_layers_to_tune} layers instead.")
    for l in layers[-min_layers_to_tune:]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj, **lora_config)
        l.self_attn.k_proj = LoRALinear.from_linear(l.self_attn.k_proj, **lora_config)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj, **lora_config)
        l.self_attn.o_proj = LoRALinear.from_linear(l.self_attn.o_proj, **lora_config)



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


def save_checkpoint(model: nn.Module, checkpoint_file_name: str, step: str, timestamp: str):
    """
    Saves the LoRA adapter weights as a .safetensors file.

    Note: The 'extension' parameter from the function call is ignored to
    ensure the correct .safetensors extension is used.
    """
    # 1. Get the trainable parameters (which are only the LoRA weights
    #    since the base model is frozen) and flatten them into a dictionary.
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    
    checkpoint_file = f"./adapters/lora_wordle/{checkpoint_file_name}_{step}_{timestamp}.safetensors"

    # 3. Save the weights using the safetensors utility.
    #    It takes the filename and the flat dictionary of weights.
    mx.savez(checkpoint_file, **adapter_weights)
    
    print(f"\nSaved LoRA adapter checkpoint to {checkpoint_file} (in safetensors format)")

def load_adapter(model: nn.Module, adapter_path: str):
    """
    Manually loads LoRA adapter weights into a model.
    This is a workaround for older mlx-lm versions that lack model.load_adapter().
    """
    print(f"Loading adapter weights from: {adapter_path}")
    try:
        # Load the dictionary of adapter weights (e.g., {'layers.0.attention.q_proj.lora_a': array(...)})
        adapter_weights = mx.load(adapter_path)
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


# ==============================================================================
# --- 2. The Merging Function ---
# ==============================================================================

def _get_module_by_path(module: nn.Module, path: str):
    """
    Access a nested module within a parent module using a dotted path string.
    """
    return reduce(getattr, path.split('.'), module)

def _set_module_by_path(parent_module: nn.Module, path: str, child_module: nn.Module):
    """
    Sets a nested module within a parent module using a dotted path string.
    This version correctly handles paths that include lists of layers (e.g., 'layers.0').
    """
    path_parts = path.split('.')
    current_module = parent_module
    
    # Traverse to the second-to-last module in the path
    for part in path_parts[:-1]:
        if part.isdigit():
            # Handle list indexing for layers
            current_module = current_module[int(part)]
        else:
            # Handle attribute access for modules
            current_module = getattr(current_module, part)
            
    # Set the final module on the direct parent
    final_part = path_parts[-1]
    if final_part.isdigit():
         current_module[int(final_part)] = child_module
    else:
        setattr(current_module, final_part, child_module)

def dequantize_model_inplace(model_to_dequantize: nn.Module):
    """
    A helper function to find all QuantizedLinear layers in a model
    and replace them with standard Linear layers in place.
    """
    for name, module in model_to_dequantize.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            # De-quantize the weights and biases
            weight = mx.dequantize(
                module.weight, module.scales, module.biases, module.group_size, module.bits
            ).astype(mx.float16)
            
            bias = hasattr(module, "bias") and module.bias is not None
            output_dims, input_dims = weight.shape
            new_linear = nn.Linear(input_dims, output_dims, bias=bias)
            new_linear.weight = weight
            if bias:
                new_linear.bias = module.bias
            
            _set_module_by_path(model_to_dequantize, name, new_linear)


def merge_lora_weights(config: cfg.TrainerConfig, adapter_path: str, output_path: str) -> tuple[nn.Module, PreTrainedTokenizerFast]:
    """
    Loads a base model, de-quantizes if necessary, applies and fuses LoRA
    weights, and saves the merged model with all necessary config files.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_id = config.model.name
    
    print(f"Loading base model: {model_id}")
    model, tokenizer = load(model_id)
    print(model)
    # --- START: MODIFIED SECTION ---
    # 1. De-quantize the model immediately after loading
    is_quantized_model = any("Quantized" in str(type(m)) for m in model.modules())
    if is_quantized_model:
        print("Quantized model detected. De-quantizing in-memory...")
        dequantize_model_inplace(model) # Use the robust de-quantization function
        print("De-quantization complete. Model is now full-precision.")
        
        # Verification Step: Check if any quantized modules remain
        is_still_quantized = any("Quantized" in str(type(m)) for m in model.modules())
        if is_still_quantized:
            print("WARNING: Model still contains quantized layers after de-quantization!")
        else:
            print("✅ Verification successful: No quantized layers remain.")
    else:
        print("Full-precision model detected. No de-quantization needed.")

    # # 1. De-quantize the model immediately after loading
    # # This is the new, correct location for this logic.
    # is_quantized_model = any(isinstance(m, nn.QuantizedLinear) for _, m in model.language_model.model.named_modules())
    # if is_quantized_model:
    #     print("Quantized model detected. De-quantizing in-memory before applying LoRA...")
    #     dequantize_model_inplace(model.language_model.model)
    #     print(model)
    #     # Manually handle the QuantizedEmbedding layer
    #     if hasattr(model.language_model.model, "embed_tokens") and isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding):
    #         print("Manually de-quantizing embedding layer...")
    #         q_emb = model.language_model.model.embed_tokens
    #         # Explicitly use the model's hidden dimension (d_model) for the embedding output
    #         print(model.language_model.model.layers[0].self_attn.q_proj.weight.shape[1])
    #         d_model = model.language_model.model.layers[0].self_attn.q_proj.weight.shape[1]
    #         new_emb = nn.Embedding(q_emb.weight.shape[0], d_model)
    #         new_emb.weight = q_emb.weight # This access de-quantizes the weights
    #         model.language_model.model.embed_tokens = new_emb
    #     print("De-quantization complete. Model is now full-precision.")
    # else:
    #     print("Full-precision model detected. No de-quantization needed.")

    # --- END: MODIFIED SECTION ---

    # 2. Apply the LoRA structure to the now full-precision model
    print("Applying LoRA structure to the model...")
    model.freeze()
    lora_config = {"rank": config.lora.rank, "alpha": config.lora.alpha}

    layers = model.language_model.model.layers # gemma 4b
    # layers = model.model.layers # gemma 1b
    layers_to_tune = min(config.lora.layers_to_tune, len(layers))
    for l in layers[-layers_to_tune:]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj, **lora_config)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj, **lora_config)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate, **lora_config)

    # 3. Load the trained adapter weights
    print(f"Loading adapter weights from: {adapter_path}")
    model.load_weights(adapter_path, strict=False)

    # The old de-quantization block that was here has been removed.

    # 4. Fuse the LoRA layers
    print("Fusing LoRA weights into the base model...")
    for l in layers:
        if hasattr(l.self_attn, "q_proj") and isinstance(l.self_attn.q_proj, LoRALinear):
            l.self_attn.q_proj = l.self_attn.q_proj.to_linear()
        if hasattr(l.self_attn, "v_proj") and isinstance(l.self_attn.v_proj, LoRALinear):
            l.self_attn.v_proj = l.self_attn.v_proj.to_linear()
        if hasattr(l, "block_sparse_moe") and isinstance(l.block_sparse_moe.gate, LoRALinear):
            l.block_sparse_moe.gate = l.block_sparse_moe.gate.to_linear()
            
    model.unfreeze()
    print("Converting model parameters to float32 for saving...")
    # This is the new line. It iterates through all model parameters (the "leaves" of the tree)
    # and converts each one to float32. This ensures compatibility with the saving library.
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

    # 5. Save the final merged model and configs
    print(f"Saving merged model to: {output_dir}")
    weights = dict(tree_flatten(model.parameters()))
    # print(weights)
    save_safetensors(weights, str(output_dir / "model.safetensors"))
    tokenizer.save_pretrained(str(output_dir))

    # Save config.json for architecture
    try:
        config_file_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_file_path, 'r') as f_in, open(output_dir / "config.json", 'w') as f_out:
            f_out.write(f_in.read())
    except Exception as e:
        print(f"WARNING: Could not save config.json: {e}")

    # Save adapter_config.json for metadata
    adapter_config = { "model": model_id, "lora_parameters": asdict(config.lora), "training_parameters": asdict(config.training) }
    with open(output_dir / "adapter_config.json", 'w') as f:
        json.dump(adapter_config, f, indent=4)
        
    print(f"\n✅ Merge complete! Your merged model is ready at: {output_dir}")
    return model, tokenizer

# ANSI escape codes for colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ==============================================================================
# --- LoRA Comparing Adapters utility  ---
# ==============================================================================
def check_if_untrained(adapter: np.lib.npyio.NpzFile) -> bool:
    """
    Checks if a LoRA adapter is in its initial, untrained state by verifying
    if all 'lora_b' weights are zero.

    Returns:
        True if the adapter appears untrained, False otherwise.
    """
    # Find all keys corresponding to lora_b weights
    lora_b_keys = [k for k in adapter.keys() if k.endswith('.lora_b')]

    if not lora_b_keys:
        # This isn't a standard LoRA file if no lora_b weights are found
        return False

    # Check if all values in all lora_b matrices are zero
    is_untrained = all(np.all(adapter[key] == 0) for key in lora_b_keys)
    
    return is_untrained
# ==============================================================================

def compare_lora_adapters(file_path1: Path, file_path2: Path):
    """
    Loads two LoRA adapter .npz files and provides a detailed comparison
    of their structure and weight differences.
    """
    print(f"{bcolors.HEADER}--- Comparing LoRA Adapters ---{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Adapter 1:{bcolors.ENDC} {file_path1}")
    print(f"{bcolors.OKCYAN}Adapter 2:{bcolors.ENDC} {file_path2}\n")

    try:
        adapter1 = np.load(file_path1)
        adapter2 = np.load(file_path2)
    except FileNotFoundError as e:
        print(f"{bcolors.FAIL}Error: Could not load file. {e}{bcolors.ENDC}")
        return

    # --- 1. Initial Diagnostics ---
    print(f"{bcolors.HEADER}--- Initial Diagnostics ---{bcolors.ENDC}")
    is_untrained1 = check_if_untrained(adapter1)
    if is_untrained1:
        print(f"{bcolors.FAIL}✖ WARNING: Adapter 1 ({file_path1.name}) appears to be in an initial, UNTRAINED state.{bcolors.ENDC}")
        print(f"{bcolors.WARNING}  > All 'lora_b' weights are zero.{bcolors.ENDC}")

    is_untrained2 = check_if_untrained(adapter2)
    if is_untrained2:
        print(f"{bcolors.FAIL}✖ WARNING: Adapter 2 ({file_path2.name}) appears to be in an initial, UNTRAINED state.{bcolors.ENDC}")
        print(f"{bcolors.WARNING}  > All 'lora_b' weights are zero.{bcolors.ENDC}")
    
    if not is_untrained1 and not is_untrained2:
        print(f"{bcolors.OKGREEN}✔ Both adapters appear to be trained (non-zero 'lora_b' weights found).{bcolors.ENDC}")
    print("")


    keys1 = set(adapter1.keys())
    keys2 = set(adapter2.keys())

    # --- 2. Structural Comparison (Key Sets) ---
    print(f"{bcolors.HEADER}--- Structural Comparison ---{bcolors.ENDC}")
    if keys1 == keys2:
        print(f"{bcolors.OKGREEN}✔ Both adapters have the exact same set of {len(keys1)} weights.{bcolors.ENDC}\n")
    else:
        print(f"{bcolors.FAIL}✖ Adapters have different structures.{bcolors.ENDC}")
        unique_to_1 = keys1 - keys2
        if unique_to_1:
            print(f"{bcolors.WARNING}  > Keys only in {file_path1.name}: {unique_to_1}{bcolors.ENDC}")
        unique_to_2 = keys2 - keys1
        if unique_to_2:
            print(f"{bcolors.WARNING}  > Keys only in {file_path2.name}: {unique_to_2}{bcolors.ENDC}")
        print("")

    # --- 3. Detailed Weight Comparison (For Common Keys) ---
    common_keys = sorted(list(keys1.intersection(keys2)))
    if not common_keys:
        print(f"{bcolors.FAIL}No common weights to compare.{bcolors.ENDC}")
        return

    print(f"{bcolors.HEADER}--- Detailed Weight Comparison ({len(common_keys)} Common Weights) ---{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{'Weight Key':<60} {'Max Abs Diff':<20} {'Mean Abs Diff':<20} {'Cosine Sim':<20}{bcolors.ENDC}")
    print("-" * 120)

    max_overall_diff = 0

    for key in common_keys:
        w1 = adapter1[key]
        w2 = adapter2[key]

        if w1.shape != w2.shape:
            print(f"{bcolors.FAIL}{key:<60} SHAPE MISMATCH! {w1.shape} vs {w2.shape}{bcolors.ENDC}")
            continue

        abs_diff = np.abs(w1 - w2)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()
        norm1 = np.linalg.norm(w1_flat)
        norm2 = np.linalg.norm(w2_flat)
        
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 1.0 if norm1 == norm2 else 0.0
        else:
            cosine_sim = np.dot(w1_flat, w2_flat) / (norm1 * norm2)

        max_overall_diff = max(max_overall_diff, max_abs_diff)
        
        if max_abs_diff < 1e-6:
            color = bcolors.OKGREEN
        elif max_abs_diff < 1e-3:
            color = bcolors.OKCYAN
        else:
            color = bcolors.WARNING

        print(f"{color}{key:<60} {max_abs_diff:<20.6e} {mean_abs_diff:<20.6e} {cosine_sim:<20.6f}{bcolors.ENDC}")

    print("-" * 120)
    print(f"{bcolors.BOLD}Comparison Summary:{bcolors.ENDC}")
    if max_overall_diff < 1e-6:
        print(f"{bcolors.OKGREEN}✔ The adapters are numerically identical.{bcolors.ENDC}")
    else:
        print(f"{bcolors.WARNING}The adapters are numerically different. Max difference observed: {max_overall_diff:.6e}{bcolors.ENDC}")
