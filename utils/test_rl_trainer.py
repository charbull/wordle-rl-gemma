import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import mlx.core as mx
import mlx.nn as nn
from datasets import Dataset

# Import the functions to be tested from your script
from utils.rl_trainer import (
    pad_sequences,
    get_named_parameters_flat,
    load_wordle_trajectories_from_jsonl,
    evaluate,
    get_log_probs,
    is_nan_or_inf,
    grpo_loss_and_grad
)

# Mocking necessary external objects
mock_rl_config = MagicMock()
mock_rl_config.evaluation.samples = 10
mock_rl_config.rl.max_trials = 6

# Mock GRPO config
mock_grpo_config = MagicMock()
mock_grpo_config.grpo.beta = 0.1
mock_grpo_config.grpo.kl_coeff = 0.02


class TestHelperFunctions(unittest.TestCase):
    """Tests for standalone helper functions."""

    def test_pad_sequences(self):
        token_lists = [[1, 2], [3, 4, 5], [6]]
        expected = mx.array([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        result = pad_sequences(token_lists, 0)
        self.assertTrue(mx.array_equal(result, expected))

    def test_get_named_parameters_flat(self):
        nested_params = {
            'layer1': {'weight': mx.array([1]), 'bias': mx.array([2])},
            'layer2': {'sub_layer': {'weight': mx.array([3])}}
        }
        flat_list = get_named_parameters_flat(nested_params)
        flat_dict = {name: val.item() for name, val in flat_list}
        expected_dict = {
            'layer1.weight': 1,
            'layer1.bias': 2,
            'layer2.sub_layer.weight': 3
        }
        self.assertDictEqual(flat_dict, expected_dict)

    def test_is_nan_or_inf(self):
        self.assertFalse(is_nan_or_inf(mx.array([1, 2, 3])))
        self.assertTrue(is_nan_or_inf(mx.array([1, float('nan'), 3])))
        self.assertTrue(is_nan_or_inf(mx.array([1, float('inf'), 3])))


class TestLoadWordleTrajectories(unittest.TestCase):
    """Tests the JSONL data loading function."""

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "data": {"secret": "TABLE", "messages": ["msg1"]}
    }) + "\n" + json.dumps({
        "data": {"secret": "CHAIR", "messages": ["msg2"]}
    }))
    def test_successful_load(self, mock_file):
        dataset = load_wordle_trajectories_from_jsonl("dummy_path.jsonl")
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['secret'], 'TABLE')

    @patch("builtins.open")
    def test_file_not_found(self, mock_open_func):
        mock_open_func.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_wordle_trajectories_from_jsonl("non_existent_path.jsonl")


class TestGetLogProbs(unittest.TestCase):
    """Tests the log probability calculation."""

    def test_log_probs_calculation(self):
        class MockModel(nn.Module):
            def __call__(self, inputs):
                batch_size, seq_len = inputs.shape
                vocab_size = 10
                logits = mx.full((batch_size, seq_len, vocab_size), -1e9) # Use a large negative number
                rows = mx.arange(batch_size).reshape(-1, 1)
                cols = mx.arange(seq_len-1)
                next_tokens = inputs[:, 1:]
                logits[rows, cols, next_tokens] = 10.0
                return logits

        model = MockModel()
        prompt_ids = mx.array([[1, 2], [1, 2]])
        output_ids = mx.array([[3, 4], [3, 0]])
        log_probs = get_log_probs(model, prompt_ids, output_ids, pad_token_id=0)
        self.assertAlmostEqual(log_probs[0].item(), 0, places=2)
        self.assertAlmostEqual(log_probs[1].item(), 0, places=2)


class TestGRPOLoss(unittest.TestCase):
    """Tests the GRPO loss function."""

    def setUp(self):
        self.prompt_toks = mx.array([[1, 2]])
        self.winner_toks = mx.array([[3, 4]])
        self.loser_toks = mx.array([[5, 6]])

    @patch('utils.rl_trainer.get_log_probs')
    def test_positive_loss_when_policy_improves(self, mock_get_log_probs):
        mock_get_log_probs.side_effect = [
            mx.array([-1.0]), mx.array([-5.0]), # policy
            mx.array([-3.0]), mx.array([-3.0]), # ref
        ]
        loss = grpo_loss_and_grad(
            {}, MagicMock(), MagicMock(), self.winner_toks, self.loser_toks, self.prompt_toks,
            mock_grpo_config, 0
        )
        self.assertGreater(loss.item(), 0)

    @patch('utils.rl_trainer.get_log_probs')
    def test_loss_when_no_change(self, mock_get_log_probs):
        """Loss should be -log(0.5) if policy and ref have same preferences."""
        mock_get_log_probs.side_effect = [
            mx.array([-1.0]), mx.array([-5.0]), # policy
            mx.array([-1.0]), mx.array([-5.0]), # ref
        ]
        config_no_kl = MagicMock()
        config_no_kl.grpo.beta = 0.1
        config_no_kl.grpo.kl_coeff = 0.0 # Isolate GPRO loss
        loss = grpo_loss_and_grad(
            {}, MagicMock(), MagicMock(), self.winner_toks, self.loser_toks, self.prompt_toks,
            config_no_kl, 0
        )
        self.assertAlmostEqual(loss.item(), -mx.log(mx.array(0.5)).item(), places=4)

    @patch('utils.rl_trainer.get_log_probs')
    def test_kl_penalty_adds_to_loss(self, mock_get_log_probs):
        mock_get_log_probs.side_effect = [
            mx.array([-0.5]), mx.array([-5.0]),  # policy
            mx.array([-2.0]), mx.array([-5.0]),  # ref
        ]
        grpo_logits = (-0.5 - -5.0) - (-2.0 - -5.0)  # 4.5 - 3.0 = 1.5
        expected_grpo_loss = -mx.log(mx.sigmoid(mock_grpo_config.grpo.beta * grpo_logits)).item()
        kl_div = -0.5 - (-2.0)  # 1.5
        expected_kl_penalty = mock_grpo_config.grpo.kl_coeff * kl_div
        expected_total_loss = expected_grpo_loss + expected_kl_penalty
        
        loss = grpo_loss_and_grad(
            {}, MagicMock(), MagicMock(), self.winner_toks, self.loser_toks, self.prompt_toks,
            mock_grpo_config, 0
        )
        self.assertAlmostEqual(loss.item(), expected_total_loss, places=4)


class TestEvaluateFunction(unittest.TestCase):
    """Tests the evaluation function by mocking the game playing."""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_dataset = Dataset.from_list([{'secret': 'TABLE'}, {'secret': 'CHAIR'}, {'secret': 'PLUMB'}])

    @patch('utils.rl_trainer.play_wordle_game')
    def test_evaluation_metrics(self, mock_play_wordle_game):
        """Should correctly calculate win rate and average turns."""
        # Game 1: Win in 3 turns
        win_game_1 = MagicMock(
            solved=True, 
            attempts=[MagicMock(prompt_string=f"turn_{i}") for i in range(3)]
        )
        # Game 2: Loss
        loss_game = MagicMock(solved=False)
        # Game 3: Win in 5 turns
        win_game_2 = MagicMock(
            solved=True, 
            attempts=[MagicMock(prompt_string=f"turn_{i}") for i in range(5)]
        )
        mock_play_wordle_game.side_effect = [win_game_1, loss_game, win_game_2]

        results = evaluate(MagicMock(), self.mock_tokenizer, self.mock_dataset, mock_rl_config, "")
        
        self.assertAlmostEqual(results['win_rate'], 2 / 3)
        self.assertAlmostEqual(results['avg_turns_on_win'], 4.0) # (3 + 5) / 2
        self.assertDictEqual(results['turn_distribution_on_win'], {3: 1, 5: 1})
        self.assertEqual(mock_play_wordle_game.call_count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)