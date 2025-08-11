import unittest
from unittest.mock import patch, mock_open, MagicMock, ANY
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from datasets import Dataset

# Import the functions to be tested from your script
# Adjust the import if your file has a different name
from utils.rl_trainer import (
    pad_sequences,
    get_named_parameters_flat,
    load_wordle_trajectories_from_jsonl,
    evaluate,
    get_log_probs,
    is_nan_or_inf
)

# Mocking necessary external objects
# A mock config object that simulates the real config
mock_config = MagicMock()
mock_config.evaluation.samples = 10
mock_config.rl.max_trials = 6

# A mock GameRollout object to simulate results from play_wordle_game
mock_game_rollout = MagicMock()

class TestHelperFunctions(unittest.TestCase):
    """Tests for standalone helper functions."""

    def test_pad_sequences(self):
        """Should pad a list of token lists to the same max length."""
        token_lists = [[1, 2], [3, 4, 5], [6]]
        pad_value = 0
        expected = mx.array([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        result = pad_sequences(token_lists, pad_value)
        self.assertTrue(mx.array_equal(result, expected))

    def test_pad_sequences_empty(self):
        """Should handle an empty list of sequences."""
        result = pad_sequences([], 0)
        self.assertEqual(result.shape, (0,))

    def test_get_named_parameters_flat(self):
        """Should correctly flatten a nested dictionary of parameters."""
        nested_params = {
            'layer1': {'weight': mx.array([1]), 'bias': mx.array([2])},
            'layer2': {'sub_layer': {'weight': mx.array([3])}}
        }
        flat_list = get_named_parameters_flat(nested_params)
        # Convert to a dictionary for easier comparison, ignoring array values for simplicity
        flat_dict = {name: val.item() for name, val in flat_list}
        
        expected_dict = {
            'layer1.weight': 1,
            'layer1.bias': 2,
            'layer2.sub_layer.weight': 3
        }
        self.assertDictEqual(flat_dict, expected_dict)

    def test_is_nan_or_inf(self):
        """Should correctly detect NaN or Infinity in an mx.array."""
        self.assertFalse(is_nan_or_inf(mx.array([1, 2, 3])))
        self.assertTrue(is_nan_or_inf(mx.array([1, float('nan'), 3])))
        self.assertTrue(is_nan_or_inf(mx.array([1, float('inf'), 3])))
        self.assertTrue(is_nan_or_inf(mx.array([1, float('-inf'), 3])))


class TestLoadWordleTrajectories(unittest.TestCase):
    """Tests the JSONL data loading function."""

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "data": {"secret": "TABLE", "messages": ["msg1"]}
    }) + "\n" + json.dumps({
        "data": {"secret": "CHAIR", "messages": ["msg2"]}
    }))
    def test_successful_load(self, mock_file):
        """Should successfully load trajectories from a valid JSONL file."""
        dataset = load_wordle_trajectories_from_jsonl("dummy_path.jsonl")
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['secret'], 'TABLE')
        self.assertEqual(dataset[1]['messages'], ['msg2'])

    @patch("builtins.open")
    def test_file_not_found(self, mock_open_func):
        """Should raise FileNotFoundError if the file doesn't exist."""
        mock_open_func.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_wordle_trajectories_from_jsonl("non_existent_path.jsonl")
            
    @patch("builtins.open", new_callable=mock_open, read_data="not a valid json")
    def test_json_decode_error(self, mock_file):
        """Should raise JSONDecodeError for malformed files."""
        with self.assertRaises(json.JSONDecodeError):
            load_wordle_trajectories_from_jsonl("bad_json.jsonl")


class TestGetLogProbs(unittest.TestCase):
    """Tests the log probability calculation."""

    def test_log_probs_calculation(self):
        """Should correctly calculate log probs and mask padding."""
        vocab_size = 10
        pad_token_id = 0
        batch_size = 2

        class MockModel(nn.Module):
            def __call__(self, inputs, mask=None):
                logits = mx.zeros((inputs.shape[0], inputs.shape[1] - 1, vocab_size))
                for i in range(inputs.shape[1] - 1):
                    next_token = inputs[:, i + 1]
                    # High logit for correct token
                    logits[mx.arange(batch_size), i, next_token] = 10.0
                return logits

        model = MockModel()
        prompt_ids = mx.array([[1, 2], [1, 2]])
        output_ids = mx.array([[3, 4], [3, pad_token_id]])
        
        log_probs = get_log_probs(model, prompt_ids, output_ids, pad_token_id)
        
        self.assertLess(
            abs(log_probs[0].item()), 1e-3,
            "Log prob for a confident prediction should be very close to zero."
        )
        self.assertLess(
            abs(log_probs[1].item()), 1e-3,
            "Log prob with padding should also be close to zero."
        )


class TestEvaluateFunction(unittest.TestCase):
    """Tests the evaluation function by mocking the game playing."""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_dataset = Dataset.from_list([{'secret': 'TABLE'}, {'secret': 'CHAIR'}, {'secret': 'PLUMB'}])

    @patch('utils.rl_trainer.play_wordle_game')
    def test_evaluation_metrics(self, mock_play_wordle_game):
        """Should correctly calculate win rate and average turns."""
        # --- Setup mock return values ---
        # Game 1: Win in 3 turns
        win_game = MagicMock()
        win_game.solved = True
        win_game.attempts = [MagicMock(prompt_string="p1"), MagicMock(prompt_string="p2"), MagicMock(prompt_string="p3")]

        # Game 2: Loss
        loss_game = MagicMock()
        loss_game.solved = False

        # Game 3: Win in 5 turns
        win_game_long = MagicMock()
        win_game_long.solved = True
        win_game_long.attempts = [MagicMock(prompt_string=f"p{i}") for i in range(5)]

        mock_play_wordle_game.side_effect = [win_game, loss_game, win_game_long]

        # --- Run evaluation ---
        results = evaluate(
            model=MagicMock(),
            tokenizer=self.mock_tokenizer,
            dataset=self.mock_dataset,
            config=mock_config,
            system_prompt=""
        )

        # --- Assertions ---
        # 2 wins out of 3 games
        self.assertAlmostEqual(results['win_rate'], 2 / 3)
        # Avg turns for the 2 wins (3 turns + 5 turns) / 2 = 4
        self.assertAlmostEqual(results['avg_turns_on_win'], 4.0)
        # Check the distribution of turns for wins
        self.assertDictEqual(results['turn_distribution_on_win'], {3: 1, 5: 1})
        # Ensure the mock was called for each sample in the dataset
        self.assertEqual(mock_play_wordle_game.call_count, 3)

    def test_empty_dataset(self):
        """Should handle an empty dataset gracefully."""
        empty_dataset = Dataset.from_list([])
        results = evaluate(MagicMock(), self.mock_tokenizer, empty_dataset, mock_config, "")
        self.assertEqual(results['win_rate'], 0.0)
        self.assertEqual(results['avg_turns_on_win'], 0.0)
        self.assertDictEqual(results['turn_distribution_on_win'], {})


if __name__ == '__main__':
    print("--- Running Unit Tests for RL Trainer Helper Functions ---")
    print("Note: The main `train` loop is not unit tested as it requires a full integration setup.")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)