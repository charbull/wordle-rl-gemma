# In utils/test_aaa_warmup.py

import unittest
import mlx.core as mx

class TestWarmup(unittest.TestCase):
    def test_mlx_initialization(self):
        """
        This is a simple 'warm-up' test. Its only purpose is to trigger the
        one-time initialization of the MLX framework by creating a single tensor.
        This absorbs the startup cost so that subsequent, more complex tests
        can run at full speed without being timed for this setup.
        """
        print("\n--- Running MLX Warm-up Test ---")
        # Creating the first tensor triggers the one-time device setup.
        a = mx.array([1, 2, 3])
        mx.eval(a) # Ensure the operation completes.
        self.assertEqual(a.shape, (3,))
        print("--- MLX Warm-up Complete ---")