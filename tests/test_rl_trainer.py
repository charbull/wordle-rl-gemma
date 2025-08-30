import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import mlx.core as mx
import mlx.nn as nn
from datasets import Dataset
from src.wordle.game import GameRollout, GameRecord, load_wordle_trajectories_from_jsonl


from src.ml.rl_trainer import (
    pad_sequences,
    get_named_parameters_flat,
    evaluate,
    get_log_probs,
    is_nan_or_inf,
    grpo_loss_and_grad
)


mock_rl_config = MagicMock()
mock_rl_config.evaluation.samples = 10
mock_rl_config.rl.max_trials = 6

mock_grpo_config = MagicMock()
mock_grpo_config.grpo.beta = 0.1
mock_grpo_config.grpo.kl_coeff = 0.02


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions."""

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
                logits = mx.full((batch_size, seq_len, vocab_size), -1e9)
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
        self.mock_config = MagicMock()
        self.mock_config.grpo.beta = 0.1
        self.mock_config.grpo.kl_coeff = 0.1
        self.prompt_toks = mx.array([[1, 2]])
        self.winner_toks = mx.array([[3, 4]])
        self.loser_toks = mx.array([[5, 6]])
        self.trainable_params = {"lora_weight": mx.array(1.0)}
        self.pad_id = 0
        self.mock_policy_model = MagicMock()
        self.mock_ref_model = MagicMock()

    @patch('src.ml.rl_trainer.get_log_probs')
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

    @patch('src.ml.rl_trainer.get_log_probs')
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

    def test_kl_penalty_adds_to_loss(self):
        """
        Verifies that adding a KL penalty increases the total loss.
        """
        # Make the policy much less confident than the reference model
        with patch('src.ml.rl_trainer.get_log_probs') as mock_get_log_probs:
            mock_get_log_probs.side_effect = [
                # First call for policy_model
                mx.array([-10.0]), # policy winner log_prob (low confidence)
                mx.array([-11.0]), # policy loser log_prob
                # Second call for ref_model
                mx.array([-5.0]),  # ref winner log_prob (high confidence)
                mx.array([-6.0])   # ref loser log_prob
            ]

            # --- Calculate loss with the KL penalty ---
            self.mock_config.grpo.kl_coeff = 0.1
            # Use the real loss function now
            loss_with_kl = grpo_loss_and_grad(
                self.trainable_params, self.mock_policy_model, self.mock_ref_model,
                self.winner_toks, self.loser_toks, self.prompt_toks,
                self.mock_config, self.pad_id
            )

        # Reset the mock for the second calculation
        with patch('src.ml.rl_trainer.get_log_probs') as mock_get_log_probs:
            mock_get_log_probs.side_effect = [
                mx.array([-10.0]), mx.array([-11.0]),
                mx.array([-5.0]),  mx.array([-6.0])
            ]
            
            # Calculate loss WITHOUT the KL penalty ---
            self.mock_config.grpo.kl_coeff = 0.0
            loss_without_kl = grpo_loss_and_grad(
                self.trainable_params, self.mock_policy_model, self.mock_ref_model,
                self.winner_toks, self.loser_toks, self.prompt_toks,
                self.mock_config, self.pad_id
            )

        self.assertGreater(loss_with_kl.item(), loss_without_kl.item(),
                         "KL penalty should increase the total loss.")


class TestEvaluateFunction(unittest.TestCase):
    """Tests the evaluation function by mocking the game playing."""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_config = MagicMock()
        self.mock_config.evaluation.samples = 3
        self.mock_tokenizer = MagicMock()
        mock_data = [
            {
                'secret': 'TABLE', 
                'messages': [
                    {'role': 'system', 'content': ''}, 
                    {'role': 'user', 'content': '...'}
                ]
            },
            {
                'secret': 'CHAIR', 
                'messages': [
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': '...'}
                ]
            },
            {
                'secret': 'PLUMB', 
                'messages': [
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': '...'}
                ]
            }
        ]
        self.mock_dataset = Dataset.from_list(mock_data)

    @patch('src.ml.rl_trainer.play_wordle_game')
    @patch('src.utils.logging.log_game_result')
    def test_evaluation_metrics(self, mock_log_game_result, mock_play_wordle_game):
        # --- 1. Setup Mocks ---
        # We need to simulate the GameRollout objects that play_wordle_game returns
        win_game_1_rollout = MagicMock(spec=GameRollout, solved=True)
        # To calculate turns, we need the attempts attribute to be a list of mocks
        win_game_1_rollout.attempts = [MagicMock(prompt_string="t1"), MagicMock(prompt_string="t2"), MagicMock(prompt_string="t3")]
        
        loss_game_rollout = MagicMock(spec=GameRollout, solved=False)
        loss_game_rollout.attempts = [] # No winning attempt
        
        win_game_2_rollout = MagicMock(spec=GameRollout, solved=True)
        win_game_2_rollout.attempts = [MagicMock(prompt_string=f"t{i}") for i in range(5)]
        
        mock_play_wordle_game.side_effect = [win_game_1_rollout, loss_game_rollout, win_game_2_rollout]

        # We also mock log_game_result to return predictable GameRecord objects (or mocks of them)
        # This decouples the test from the implementation details of log_game_result
        record1 = MagicMock(spec=GameRecord, solved=True, turns_to_solve=3)
        record2 = MagicMock(spec=GameRecord, solved=False, turns_to_solve=-1)
        record3 = MagicMock(spec=GameRecord, solved=True, turns_to_solve=5)
        mock_log_game_result.side_effect = [record1, record2, record3]

        # The function now returns a list of these GameRecord-like mock objects
        results_list = evaluate(MagicMock(), self.mock_tokenizer, self.mock_dataset, self.mock_config, "", 150)
        
        # The primary result is a list of 3 records
        self.assertEqual(len(results_list), 3)
        self.assertEqual(mock_play_wordle_game.call_count, 3)
        
        # Now, we calculate the stats from the list to verify the data is correct
        wins = sum(1 for r in results_list if r.solved)
        total_games = len(results_list)
        win_rate = wins / total_games
        
        win_turns = [r.turns_to_solve for r in results_list if r.solved]
        avg_turns_on_win = sum(win_turns) / len(win_turns) if win_turns else 0.0

        self.assertEqual(wins, 2)
        self.assertAlmostEqual(win_rate, 2 / 3)
        self.assertAlmostEqual(avg_turns_on_win, 4.0) # (3 + 5) / 2


if __name__ == '__main__':
    unittest.main(verbosity=2)