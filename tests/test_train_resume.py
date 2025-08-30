import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import mlx.core as mx
from src.ml import rl_trainer
from src.utils import config as cfg
from src.wordle.game import GameRollout
from datasets import Dataset

@patch('src.ml.rl_trainer.plot_cumulative_wins')
@patch('src.ml.rl_trainer.write_metrics_to_file')
@patch('src.ml.rl_trainer.log_metrics_to_tensorboard')
@patch('src.ml.rl_trainer.prepare_data')
@patch('src.ml.rl_trainer.truncate_jsonl_log')
@patch('src.ml.rl_trainer.evaluate')
@patch('src.ml.rl_trainer.plot_training_curves')
@patch('src.ml.rl_trainer.play_wordle_game')
@patch('src.ml.rl_trainer.SummaryWriter')
@patch('src.ml.rl_trainer.lora')
@patch('src.ml.rl_trainer.load')
class TestTrainerResumeLogic(unittest.TestCase):
    def test_full_resume_workflow(
        self,
        mock_load_model,
        mock_lora,
        mock_summary_writer,
        mock_play_game,
        mock_plot,
        mock_evaluate,
        mock_truncate_log,
        mock_prepare_data,
        mock_log_tensorboard,
        mock_write_metrics,
        mock_plot_wins
    ):
        """
        Integration test to verify the trainer's resume logic.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            
            # --- SETUP for directories and checkpoint ---
            timestamp = "20250101-120000"
            run_name = f"{timestamp}_mock-model_rank16"
            run_dir = base_dir / "experiments" / run_name
            adapter_dir = run_dir / "adapters"
            tensorboard_dir = run_dir / "tensorboard"
            for d in [adapter_dir, tensorboard_dir]:
                d.mkdir(parents=True)

            resume_step = 50
            total_iterations = 75
            eval_steps = 25
            
            checkpoint_name = f"adapter_step_{resume_step}.npz"
            checkpoint_file = adapter_dir / checkpoint_name
            checkpoint_file.touch()

            dummy_data_path = base_dir / "dummy_data.jsonl"
            dummy_data_path.touch()

            dummy_data_path = base_dir / "dummy_data.jsonl"
            with open(dummy_data_path, "w") as f:
                f.write('{}\n') # Write a single line to make it a non-empty file

            # Create a real config object that points to the dummy file
            config = cfg.TrainerConfig(
                training=cfg.TrainingConfig(
                    resume_from_checkpoint=str(checkpoint_file),
                    iterations=total_iterations,
                    learning_rate=1e-4,
                    use_lr_scheduler=False,
                    lr_min=1e-5,
                    lr_decay_steps=75,
                    batch_size=8,
                    log_steps=10,
                    checkpoint_steps=25,
                    config_file=str(run_dir / "config.json"),
                    data_path=dummy_data_path,
                ),
                evaluation=cfg.EvalConfig(steps=eval_steps, samples=10),
                model=cfg.ModelConfig(name="mock-model"),
                lora=cfg.LoRAConfig(rank=16, alpha=16, dropout=0.1, layers_to_tune=16),
                rl=cfg.RLConfig(sampling_temperature=0.9, num_generations=2),
            )
            config.config_file = str(run_dir / "config.yaml")

            # Mock game rollout to enable loss calculation
            mock_attempt_1 = MagicMock(training_reward=10.0, response_tokens=[1,2,3], prompt_tokens=[0])
            mock_attempt_2 = MagicMock(training_reward=-5.0, response_tokens=[4,5,6], prompt_tokens=[0])
            mock_game_rollout = MagicMock(spec=GameRollout, solved=True, attempts=[mock_attempt_1, mock_attempt_2])
            mock_play_game.return_value = mock_game_rollout

            # Mock model and data loading
            mock_policy_model = MagicMock()
            mock_policy_model.parameters.return_value = {'layer.lora_a': mx.zeros([1])}
            mock_load_model.return_value = (mock_policy_model, MagicMock())
            mock_lora.apply_lora_to_model.return_value = mock_policy_model
            mock_lora.load_adapter.return_value = mock_policy_model

            mock_train_ds = Dataset.from_list([{'secret': 'MOCK', 'messages': [{'role':'system'}, {'content':''}]}] * 100)
            mock_val_ds = Dataset.from_list([{'secret': 'MOCK', 'messages': [{'role':'system'}, {'content':''}]}] * 100)
            mock_test_ds = Dataset.from_list([{'secret': 'MOCK', 'messages': [{'role':'system'}, {'content':''}]}] * 100)
            
            # Make the mock function return a tuple of these three objects
            mock_prepare_data.return_value = (mock_train_ds, mock_val_ds, mock_test_ds)

            # Make the mocked evaluate function return an empty list to avoid downstream errors
            mock_evaluate.return_value = []

            rl_trainer.train(config=config, system_prompt="Test")

            # 1. Verify resume setup was triggered. This remains the same.
            # The log should be truncated AT the completed step.
            mock_truncate_log.assert_called_once_with(Path(run_dir / "training_metrics.jsonl"), resume_step)
            mock_lora.load_adapter.assert_called_once()

            # 2. Verify the training loop ran for the new, correct number of steps
            expected_iterations = total_iterations - (resume_step + 1) + 1 # 1 (start at 1) + 75 - (50 + 1) + 1 (start) = 25
            self.assertEqual(
                mock_play_game.call_count,
                expected_iterations,
                f"Training loop should have run {expected_iterations} times, but ran {mock_play_game.call_count} times."
            )

            # 3. Evaluation logic assertions remain the same
            expected_eval_calls = 1 # Only at step 75
            self.assertEqual(mock_evaluate.call_count, expected_eval_calls)
            self.assertEqual(mock_evaluate.call_args.args[5], 75)

if __name__ == '__main__':
    unittest.main()