import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import mlx.core as mx
import mlx.nn as nn
from datasets import Dataset

from src.wordle.game import GameRollout, GameRecord
from src.utils.config import TrainerConfig, GSPOConfig, GRPOConfig
from src.ml.base_trainer import BaseTrainer, pad_sequences, get_log_probs, is_nan_or_inf
from src.ml.grpo_trainer import grpo_loss_and_grad
from src.ml.gspo_trainer import gspo_loss_and_grad

# 1. Create the main mock object
mock_config = MagicMock(spec=TrainerConfig)

# 2. Create mock objects for the configurations
mock_config.evaluation = MagicMock()
mock_config.rl = MagicMock()
mock_config.grpo = MagicMock(spec=GRPOConfig)
mock_config.gspo = MagicMock(spec=GSPOConfig)

# 3. Now, assign the values to the nested mock objects
mock_config.evaluation.samples = 3
mock_config.rl.max_trials = 6
mock_config.grpo.beta = 0.1
mock_config.grpo.kl_coeff = 0.02
mock_config.gspo.clip_epsilon = 0.01
mock_config.gspo.advantage_epsilon = 1e-8


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions, now likely in base_trainer."""

    def test_pad_sequences(self):
        token_lists = [[1, 2], [3, 4, 5], [6]]
        expected = mx.array([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        result = pad_sequences(token_lists, 0)
        self.assertTrue(mx.array_equal(result, expected))

    def test_is_nan_or_inf(self):
        self.assertFalse(is_nan_or_inf(mx.array([1, 2, 3])))
        self.assertTrue(is_nan_or_inf(mx.array([1, float('nan'), 3])))
        self.assertTrue(is_nan_or_inf(mx.array([1, float('inf'), 3])))

class TestLoadWordleTrajectories(unittest.TestCase):
    """Tests the JSONL data loading function."""
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(
        {"data": {"secret": "TABLE", "messages": ["msg1"]}}
    ) + "\n" + json.dumps(
        {"data": {"secret": "CHAIR", "messages": ["msg2"]}}
    ))
    def test_successful_load(self, mock_file):
        from src.wordle.game import load_wordle_trajectories_from_jsonl
        dataset = load_wordle_trajectories_from_jsonl("dummy_path.jsonl")
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['secret'], 'TABLE')

class TestGetLogProbs(unittest.TestCase):
    """Tests the log probability calculation."""
    def test_log_probs_calculation(self):
        class MockModel(nn.Module):
            def __call__(self, inputs, **kwargs):
                batch_size, seq_len = inputs.shape
                vocab_size = 10
                logits = mx.ones((batch_size, seq_len, vocab_size)) * -1e9
                for i in range(batch_size):
                    for j in range(seq_len - 1):
                        # The output at position j determines the logit for the input at j
                        correct_next_token = inputs[i, j + 1]
                        logits[i, j, correct_next_token] = 10.0
                # Return only logits, as the cache is not needed for this mock
                return logits, None

        model = MockModel()
        prompt_ids = mx.array([[1, 2], [1, 2]])
        output_ids = mx.array([[3, 4], [3, 0]])
        
        log_probs = get_log_probs(model, prompt_ids, output_ids, pad_token_id=0)
        
        self.assertAlmostEqual(log_probs[0].item(), 0.0, places=2)
        self.assertAlmostEqual(log_probs[1].item(), 0.0, places=2)

class TestGRPOLoss(unittest.TestCase):
    """Tests the GRPO loss function."""
    def setUp(self):
        self.prompt_toks = mx.array([[1, 2]])
        self.winner_toks = mx.array([[3, 4]])
        self.loser_toks = mx.array([[5, 6]])
        self.pad_id = 0

    def test_loss_is_lower_for_better_policy(self):
        """
        Verify that an improved policy results in a lower loss than a worsened policy.
        The loss itself is always non-negative. A 'reward signal' for a minimizer
        is a lower loss value.
        """
        # --- SCENARIO 1: GOOD policy (prefers the winner strongly) ---
        with patch('src.ml.grpo_trainer.get_log_probs') as mock_get_log_probs_good:
            mock_get_log_probs_good.side_effect = [
                mx.array([-1.0]), mx.array([-5.0]),  # policy (winner, loser), diff = +4
                mx.array([-3.0]), mx.array([-3.0]),  # ref (winner, loser), diff = 0
            ]
            loss_good = grpo_loss_and_grad(
                {}, MagicMock(), MagicMock(), self.winner_toks, self.loser_toks,
                self.prompt_toks, mock_config, 0
            )

        # --- SCENARIO 2: BAD policy (prefers the loser strongly) ---
        with patch('src.ml.grpo_trainer.get_log_probs') as mock_get_log_probs_bad:
            mock_get_log_probs_bad.side_effect = [
                mx.array([-5.0]), mx.array([-1.0]),  # policy (winner, loser), diff = -4
                mx.array([-3.0]), mx.array([-3.0]),  # ref (winner, loser), diff = 0
            ]
            loss_bad = grpo_loss_and_grad(
                {}, MagicMock(), MagicMock(), self.winner_toks, self.loser_toks,
                self.prompt_toks, mock_config, 0
            )

        # The loss for the good policy should be lower than the loss for the bad one.
        self.assertLess(loss_good.item(), loss_bad.item())

        # You can also verify the actual value from your run
        self.assertAlmostEqual(loss_good.item(), 0.5130, places=4)

class TestGSPOLoss(unittest.TestCase):
    """Tests the new GSPO loss function."""
    def setUp(self):
        self.prompt_toks = mx.array([[1, 2], [1, 2]])
        self.response_toks = mx.array([[3, 4], [5, 6]])
        self.pad_id = 0

    @patch('src.ml.gspo_trainer.get_log_probs')
    def test_loss_sign_with_advantage(self, mock_get_log_probs):
        rewards_pos = mx.array([0.8, 0.2])
        mock_get_log_probs.side_effect = [mx.array([-1.0, -5.0]), mx.array([-3.0, -3.0])]
        loss_pos = gspo_loss_and_grad({}, MagicMock(), MagicMock(), self.prompt_toks, self.response_toks, rewards_pos, mock_config, self.pad_id)
        self.assertLess(loss_pos.item(), 0, "Positive advantage should result in a negative loss.")

        rewards_neg = mx.array([0.2, 0.8])
        mock_get_log_probs.side_effect = [mx.array([-1.0, -5.0]), mx.array([-3.0, -3.0])]
        loss_neg = gspo_loss_and_grad({}, MagicMock(), MagicMock(), self.prompt_toks, self.response_toks, rewards_neg, mock_config, self.pad_id)
        self.assertGreater(loss_neg.item(), 0, "Negative advantage should result in a positive loss.")

    @patch('src.ml.gspo_trainer.get_log_probs')
    def test_zero_loss_with_zero_advantage(self, mock_get_log_probs):
        rewards_zero = mx.array([0.5, 0.5])
        mock_get_log_probs.side_effect = [mx.array([-1.0, -5.0]), mx.array([-3.0, -3.0])]
        loss = gspo_loss_and_grad({}, MagicMock(), MagicMock(), self.prompt_toks, self.response_toks, rewards_zero, mock_config, self.pad_id)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    @patch('src.ml.gspo_trainer.get_log_probs')
    def test_clipping_for_positive_advantage(self, mock_get_log_probs):
        rewards = mx.array([1.0, 0.0])
        advantages = (rewards - mx.mean(rewards)) / (mx.std(rewards) + 1e-8)
        mock_get_log_probs.side_effect = [mx.array([-1.0, -10.0]), mx.array([-10.0, -1.0])]
        loss = gspo_loss_and_grad({}, MagicMock(), MagicMock(), self.prompt_toks, self.response_toks, rewards, mock_config, self.pad_id)
        clipped_ratio = 1 + mock_config.gspo.clip_epsilon
        expected_term1 = clipped_ratio * advantages[0]
        unclipped_ratio_2 = mx.exp((mx.array(-10.0) - mx.array(-1.0)) / 2)
        expected_term2 = unclipped_ratio_2 * advantages[1]
        expected_loss = -mx.mean(mx.array([expected_term1, expected_term2])).item()
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)

class TestEvaluateFunction(unittest.TestCase):
    """Tests the evaluation method by mocking the game playing."""
    def setUp(self):
        class DummyTrainer(BaseTrainer):
            def _get_loss_and_grad_fn(self): pass
            def _prepare_and_compute_loss(self, game_rollout): pass
        
        with patch.object(BaseTrainer, '_setup_run'), \
             patch.object(BaseTrainer, '_load_models_and_tokenizer'), \
             patch.object(BaseTrainer, '_setup_optimizer_and_params'):
            self.trainer = DummyTrainer(mock_config, "")
        
        self.trainer.policy_model = MagicMock()
        self.trainer.tokenizer = MagicMock()
        self.trainer.config = mock_config
        self.trainer.step_counter = 150
        mock_data = [{'secret': 'TABLE', 'messages': [{}, {'content': ''}]},
                     {'secret': 'CHAIR', 'messages': [{}, {'content': ''}]},
                     {'secret': 'PLUMB', 'messages': [{}, {'content': ''}]}]
        self.trainer.validation_dataset = Dataset.from_list(mock_data)

    @patch('src.ml.base_trainer.play_wordle_game')
    @patch('src.ml.base_trainer.log_game_result')
    def test_evaluation_metrics(self, mock_log_game_result, mock_play_wordle_game):
        win_game_1 = MagicMock(spec=GameRollout, solved=True)
        loss_game = MagicMock(spec=GameRollout, solved=False)
        win_game_2 = MagicMock(spec=GameRollout, solved=True)
        mock_play_wordle_game.side_effect = [win_game_1, loss_game, win_game_2]
        record1 = MagicMock(spec=GameRecord, solved=True, turns_to_solve=3)
        record2 = MagicMock(spec=GameRecord, solved=False, turns_to_solve=-1)
        record3 = MagicMock(spec=GameRecord, solved=True, turns_to_solve=5)
        mock_log_game_result.side_effect = [record1, record2, record3]

        results_list = self.trainer.evaluate()
        
        self.assertEqual(len(results_list), 3)
        self.assertEqual(mock_play_wordle_game.call_count, 3)
        wins = sum(1 for r in results_list if r.solved)
        self.assertEqual(wins, 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)