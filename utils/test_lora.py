import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import json
import os

# Adjust import paths if needed
from utils.lora import (
    LoRALinear,
    apply_lora_to_model,
    save_adapter
)

# Mock config object for tests that need it
mock_lora_config = MagicMock()
mock_lora_config.rank = 4
mock_lora_config.alpha = 8.0
mock_lora_config.dropout = 0.1
mock_lora_config.layers_to_tune = ["q_proj", "v_proj"]


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

    @patch("pathlib.Path.mkdir")
    @patch("mlx.core.save_safetensors") 
    @patch("builtins.open", new_callable=mock_open)
    def test_save_adapter(self, mock_file_open, mock_mx_save, mock_mkdir):
        """Should save weights and config without hanging."""
        class SimpleMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)
        
        mock_model = SimpleMockModel()
        output_dir = "/tmp/my-test-adapter"
        model_name = "gemma-2b"
        
        save_adapter(mock_model, output_dir, mock_lora_config, model_name)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        expected_weights_path = os.path.join(output_dir, "adapter_model.safetensors")

        # Assert that the correct, low-level save function was called
        mock_mx_save.assert_called_once()
        # Check the filename it was called with (it's the first argument)
        self.assertEqual(mock_mx_save.call_args[0][0], expected_weights_path)

        expected_config_path = os.path.join(output_dir, "adapter_config.json")
        mock_file_open.assert_called_once_with(expected_config_path, "w")


    def test_apply_lora_does_not_hang(self):
        """Should apply LoRA to a mock model without hanging."""
        # This is the corrected mock structure that prevents infinite loops
        class MockAttention(nn.Module):
            def __init__(self): super().__init__(); self.q_proj = nn.Linear(16, 16); self.k_proj = nn.Linear(16, 16); self.v_proj = nn.Linear(16, 16); self.o_proj = nn.Linear(16, 16)
        class MockLayer(nn.Module):
            def __init__(self): super().__init__(); self.self_attn = MockAttention()
        class MockModel(nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.model = nn.Module()
                self.language_model.model.layers = [MockLayer() for _ in range(num_layers)]
                
        model = MockModel(num_layers=4)
        lora_config = {"rank": 4, "alpha": 8, "dropout": 0.0}
        
        # This call should now complete without hanging
        tuned_model = apply_lora_to_model(model, lora_config, layers_to_tune=2)
        
        self.assertIsInstance(tuned_model.language_model.model.layers[-1].self_attn.q_proj, LoRALinear)
        self.assertNotIsInstance(tuned_model.language_model.model.layers[0].self_attn.q_proj, LoRALinear)

if __name__ == '__main__':
    unittest.main()