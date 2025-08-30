import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import tempfile
from src.utils import config as cfg

from src.ml.lora import (
    LoRALinear,
    apply_lora_to_model,
    save_adapter,
    load_adapter
)


mock_save_config = MagicMock()
mock_save_config.rank = 4
mock_save_config.alpha = 8.0
mock_save_config.dropout = 0.1
mock_save_config.layers_to_tune = 16


class TestLoRALinear(unittest.TestCase):
    """Tests the mathematical correctness of the LoRALinear layer."""

    def setUp(self):
        self.input_dims, self.output_dims = 32, 64
        self.rank, self.alpha = 8, 16.0
        self.base_linear = nn.Linear(self.input_dims, self.output_dims)

    def test_initialization(self):
        lora_layer = LoRALinear(self.input_dims, self.output_dims, lora_rank=self.rank, lora_alpha=self.alpha)
        self.assertEqual(lora_layer.lora_a.shape, (self.input_dims, self.rank))
        self.assertEqual(lora_layer.lora_b.shape, (self.rank, self.output_dims))
        self.assertAlmostEqual(lora_layer.scale, self.alpha / np.sqrt(self.rank))

    def test_forward_pass_correctness(self):
        lora_layer = LoRALinear.from_linear(self.base_linear, rank=self.rank, alpha=self.alpha)
        x = mx.random.normal(shape=(1, self.input_dims))
        y_base = lora_layer.linear(x)
        lora_update = (x @ lora_layer.lora_a) @ lora_layer.lora_b
        expected_y = y_base + lora_layer.scale * lora_update
        self.assertTrue(mx.allclose(lora_layer(x), expected_y, atol=1e-6))

    def test_to_linear_merging(self):
        lora_layer = LoRALinear.from_linear(self.base_linear, rank=self.rank, alpha=self.alpha)
        lora_delta_w = (lora_layer.scale * lora_layer.lora_b.T) @ lora_layer.lora_a.T
        expected_weight = lora_layer.linear.weight + lora_delta_w.astype(lora_layer.linear.weight.dtype)
        fused_linear = lora_layer.to_linear()
        self.assertIsInstance(fused_linear, nn.Linear)
        self.assertNotIsInstance(fused_linear, LoRALinear)
        self.assertTrue(mx.allclose(fused_linear.weight, expected_weight, atol=1e-6))


class TestLoRAUtilities(unittest.TestCase):
    """Tests the utility functions for applying and saving LoRA adapters."""

    def setUp(self):
        """Set up a mock model structure for use in tests."""
        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.k_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)
                self.o_proj = nn.Linear(16, 16)
        
        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = MockAttention()
        
        class MockModel(nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.model = nn.Module()
                self.language_model.model.layers = [MockLayer() for _ in range(num_layers)]

        self.MockModel = MockModel

    @patch("pathlib.Path.mkdir")
    @patch("mlx.nn.Module.save_weights")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_adapter(self, mock_file_open, mock_save_weights, mock_mkdir):
        mock_model = self.MockModel(num_layers=2)
        output_dir = "/tmp/my-test-adapter"
        
        # Call the function that writes to the file
        save_adapter(mock_model, output_dir, mock_save_config, "gemma-2b")

        # --- Assertions about file paths and directories ---
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        expected_weights_path = os.path.join(output_dir, "adapter_model.safetensors")
        mock_save_weights.assert_called_once_with(expected_weights_path)

        expected_config_path = os.path.join(output_dir, "adapter_config.json")
        mock_file_open.assert_called_once_with(expected_config_path, "w")
        
        # 1. Get a handle to the mock file object that was opened.
        file_handle = mock_file_open()
        
        # 2. Get the list of all arguments from all .write() calls.
        write_calls = file_handle.write.call_args_list
        
        # 3. Join them together to reconstruct the full string that was written.
        written_string = "".join(call[0][0] for call in write_calls)
        
        # 4. Now, load the reconstructed JSON string.
        written_data = json.loads(written_string)
        
        # Assert that the integer value from the mock config was written
        self.assertEqual(written_data['target_modules'], 16)

    def test_apply_lora_to_model(self):
        model = self.MockModel(num_layers=4)
        config = cfg.LoRAConfig(rank=4, alpha=8, dropout=0.0, layers_to_tune=2)
        tuned_model = apply_lora_to_model(model, config)
        
        self.assertIsInstance(tuned_model.language_model.model.layers[-1].self_attn.q_proj, LoRALinear)
        self.assertIsInstance(tuned_model.language_model.model.layers[-2].self_attn.q_proj, LoRALinear)
        self.assertNotIsInstance(tuned_model.language_model.model.layers[0].self_attn.q_proj, LoRALinear)
        self.assertNotIsInstance(tuned_model.language_model.model.layers[1].self_attn.q_proj, LoRALinear)

    def test_load_adapter(self):
        """Should correctly load adapter weights and update the model."""
        # 1. Define the LoRA parameters needed to create a model.
        config = cfg.LoRAConfig(rank=4, alpha=8, dropout=0.0, layers_to_tune=1)

        
        # 2. Define a known set of adapter weights to save.
        # This dictionary represents the data we expect to load.
        known_weights = {
            "language_model.model.layers.0.self_attn.q_proj.lora_a": mx.ones((16, 4))
        }

        # 3. Use a temporary file to save and then load from.
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            adapter_path = tmp.name
            mx.savez(adapter_path, **known_weights)

            # 4. Create a fresh model that will have the adapter loaded into it.
            fresh_model = self.MockModel(num_layers=1)
            fresh_lora_model = apply_lora_to_model(fresh_model, config)
            
            # 5. Sanity check: Ensure the weight is different *before* loading.
            original_weight = fresh_lora_model.language_model.model.layers[0].self_attn.q_proj.lora_a
            self.assertFalse(mx.all(original_weight == 1.0).item())

            # 6. Call the function under test.
            load_adapter(fresh_lora_model, adapter_path)
            
            # 7. Assert that the model's parameter has been successfully updated.
            updated_weight = fresh_lora_model.language_model.model.layers[0].self_attn.q_proj.lora_a
            self.assertTrue(mx.all(updated_weight == 1.0).item())


if __name__ == '__main__':
    unittest.main(verbosity=2)