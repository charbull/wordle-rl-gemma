import unittest
import tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from src.ml import lora
from src.utils import config as cfg

TEST_MODEL = "mlx-community/gemma-3-4b-it-bf16" 


def get_lora_params(model: nn.Module) -> dict:
    """Extracts only the LoRA parameters from a model."""
    return {k: v for k, v in tree_flatten(model.parameters()) if "lora" in k}

def compare_params(params1: dict, params2: dict) -> bool:
    """Returns True if two dictionaries of MLX parameters are identical."""
    if params1.keys() != params2.keys():
        print("Parameter keys do not match.")
        return False
    for key in params1:
        if not mx.all(mx.equal(params1[key], params2[key])):
            print(f"Parameter mismatch for key: {key}")
            return False
    return True

class TestCheckpointResumeLogic(unittest.TestCase):

    def test_save_and_resume_workflow(self):
        """
        Verifies the full cycle of training, saving, loading, and resuming,
        accounting for random weight initialization.
        """
        config = cfg.LoRAConfig(rank=8, alpha=16, dropout=0.0, layers_to_tune=16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)
            

            print("\nStep 1: Setting up initial model...")
            model_to_train, _ = load(TEST_MODEL)
            model_to_train = lora.apply_lora_to_model(model_to_train, config)
            initial_lora_params = get_lora_params(model_to_train)
            self.assertTrue(len(initial_lora_params) > 0, "LoRA layers were not applied correctly.")

            print("Step 2: Simulating a training step...")
            optimizer = optim.AdamW(learning_rate=1e-4)
            fake_grads = {k: mx.random.normal(v.shape) for k, v in initial_lora_params.items()}

            # Log gradient information for the specific parameter
            if "language_model.model.layers.18.self_attn.q_proj.lora_a" in fake_grads:
                print("Gradient for 'language_model.model.layers.18.self_attn.q_proj.lora_a':", 
                      fake_grads["language_model.model.layers.18.self_attn.q_proj.lora_a"].sum())
            else:
                print("Parameter 'language_model.model.layers.18.self_attn.q_proj.lora_a' not found in gradients.")

            updated_params = optimizer.apply_gradients(fake_grads, initial_lora_params)
            model_to_train.update(tree_unflatten(list(updated_params.items())))
            mx.eval(model_to_train.parameters())

            print("Step 3: Verifying that model weights have changed...")
            trained_lora_params = get_lora_params(model_to_train)
            self.assertFalse(compare_params(initial_lora_params, trained_lora_params),
                             "Model weights should be different after a training step.")

            print(f"Step 4: Saving checkpoint to directory {save_dir}...")
            checkpoint_name = "test_adapter"
            step = "1"
            timestamp = "test_time"
            checkpoint_path = save_dir/f"{checkpoint_name}_{step}_{timestamp}.npz"
            
            lora.save_checkpoint(
                model=model_to_train, 
                checkpoint_file=str(checkpoint_path)
            )
            
            expected_checkpoint_path = save_dir / f"{checkpoint_name}_{step}_{timestamp}.npz"
            self.assertTrue(expected_checkpoint_path.is_file(), "Checkpoint file was not created.")

            print("Step 5: Loading checkpoint into a new model...")
            model_to_resume, _ = load(TEST_MODEL)
            model_to_resume = lora.apply_lora_to_model(model_to_resume, config)
            
            model_to_resume = lora.load_adapter(model=model_to_resume, adapter_path=str(expected_checkpoint_path))


            print("Step 6: Verifying that resumed model weights match trained weights...")
            resumed_lora_params = get_lora_params(model_to_resume)
            self.assertTrue(compare_params(trained_lora_params, resumed_lora_params),
                            "Resumed model's weights must exactly match the saved trained weights.")
            
if __name__ == '__main__':
    unittest.main()