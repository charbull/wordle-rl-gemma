# src/ml/base_trainer.py

import abc
import itertools
import gc
import json
import re
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.utils import config as cfg
from src.ml import lora
from src.wordle.game import GameRecord, play_wordle_game, prepare_data
from src.wordle import rewards
from src.utils.logging import log_game_result, write_metrics_to_file, log_metrics_to_tensorboard, truncate_jsonl_log, plot_training_curves, plot_cumulative_wins

# ==============================================================================
# ---  HELPER FUNCTIONS (Consolidated here) ---
# ==============================================================================

def from_trainable_parameters(model_instance: nn.Module, trainable_params: dict):
    model_instance.update(tree_unflatten(list(trainable_params.items())))
    return model_instance

def pad_sequences(token_lists: List[List[int]], pad_value: int) -> mx.array:
    if not token_lists:
        return mx.array([], dtype=mx.int32)
    max_len = max(len(tokens) for tokens in token_lists)
    padded_lists = [tokens + [pad_value] * (max_len - len(tokens)) for tokens in token_lists]
    return mx.array(padded_lists)

def is_nan_or_inf(x):
    return mx.isnan(x).any().item() or mx.isinf(x).any().item()

def get_log_probs(model: nn.Module, input_ids: mx.array, output_ids: mx.array, pad_token_id: int) -> mx.array:
    full_sequence = mx.concatenate([input_ids, output_ids], axis=1)
    
    output = model(full_sequence, cache=None)
    if isinstance(output, tuple):
        logits, _ = output
    else:
        logits = output
    
    prompt_len = input_ids.shape[1]
    gen_len = output_ids.shape[1]

    output_logits = logits[:, prompt_len - 1:-1, :]

    if output_logits.shape[1] != gen_len:
         raise ValueError(f"BUG: Logits shape ({output_logits.shape}) doesn't match response shape ({output_ids.shape}).")

    log_probabilities_stable = nn.log_softmax(output_logits, axis=-1)

    chosen_token_log_probs = mx.take_along_axis(
        log_probabilities_stable, output_ids[..., None], axis=-1
    ).squeeze(-1)
    
    mask = (output_ids != pad_token_id).astype(chosen_token_log_probs.dtype)
    masked_log_probs = chosen_token_log_probs * mask
    total_log_prob = mx.sum(masked_log_probs, axis=-1)
    
    return total_log_prob

def cosine_decay_lr(step: int, initial_lr: float, min_lr: float, decay_steps: int) -> float:
    if step >= decay_steps: return min_lr
    decay_ratio = step / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (initial_lr - min_lr)

def check_early_stopping(recent_win_rates: Deque[float], recent_rewards: Deque[float], current_win_rate: float, current_avg_reward: float, patience: int) -> bool:
    recent_win_rates.append(round(current_win_rate, 2))
    recent_rewards.append(round(current_avg_reward, 2))
    should_stop = (len(recent_win_rates) == patience and len(set(recent_win_rates)) == 1 and len(recent_rewards) == patience and len(set(recent_rewards)) == 1)
    if should_stop:
        print("\n" + "="*80 + "\nEARLY STOPPING TRIGGERED:" + f"\nWin rate stable at {recent_win_rates[0]}%." + f"\nAvg reward stable at {recent_rewards[0]}.\n" + "="*80)
        return True
    return False

def rehydrate_win_tracker(log_file: Path, maxlen: int) -> deque:
    win_tracker = deque(maxlen=maxlen)
    if not log_file.exists(): return win_tracker
    print(f"Re-hydrating win rate tracker from {log_file}...")
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get('log_type') == 'train': records.append(record)
            except json.JSONDecodeError: continue
    records.sort(key=lambda r: r.get('step', 0))
    for record in records[-maxlen:]:
        win_tracker.append(1 if record.get('solved', False) else 0)
    if win_tracker:
        initial_win_rate = (sum(win_tracker) / len(win_tracker)) * 100
        print(f"Win rate tracker re-hydrated. Initial rolling win rate: {initial_win_rate:.1f}%")
    return win_tracker


class BaseTrainer(abc.ABC):
    """Abstract Base Class for Reinforcement Learning Trainers."""
    def __init__(self, config: cfg.TrainerConfig, system_prompt: str):
        self.config = config
        self.system_prompt = system_prompt
        self.step_counter = 1
        self._setup_run()
        self._load_models_and_tokenizer()
        self._setup_optimizer_and_params()
        self.loss_and_grad_fn = self._get_loss_and_grad_fn()

    @abc.abstractmethod
    def _get_loss_and_grad_fn(self):
        pass

    @abc.abstractmethod
    def _prepare_and_compute_loss(self, game_rollout: GameRecord) -> Tuple[float, Dict[str, mx.array]]:
        pass

    def _setup_run(self):
        if self.config.training.resume_from_checkpoint:
            checkpoint_path = Path(self.config.training.resume_from_checkpoint)
            if not checkpoint_path.is_file(): raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
            self.run_dir = checkpoint_path.parent.parent
            match = re.search(r'_(\d+)\.npz$', checkpoint_path.name)
            if match: self.step_counter = int(match.group(1)) + 1
            else: raise ValueError(f"Could not parse step number from: {checkpoint_path.name}")
            print(f"Resuming run from {self.run_dir} at step {self.step_counter}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name_short = self.config.model.name.split('/')[-1]
            run_name = f"{timestamp}_{model_name_short}_rank{self.config.lora.rank}"
            self.run_dir = Path("experiments") / run_name
            print(f"Starting new run in: {self.run_dir}")
        
        self.adapter_dir = self.run_dir / "adapters"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.plots_dir = self.run_dir / "plots"
        self.metrics_file_path = self.run_dir / "training_metrics.jsonl"
        for d in [self.adapter_dir, self.tensorboard_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        if not self.config.training.resume_from_checkpoint:
            cfg.save_config(self.config, self.run_dir / self.config.training.config_file)
        
        self.writer = SummaryWriter(str(self.tensorboard_dir))
        print(f"TensorBoard logs: {self.tensorboard_dir}")
        print(f"Game metrics log: {self.metrics_file_path}")

    def _load_models_and_tokenizer(self):
        print(f"Loading base model: {self.config.model.name}")
        self.policy_model, self.tokenizer = load(self.config.model.name)
        self.ref_model, _ = load(self.config.model.name)

        self.policy_model = lora.apply_lora_to_model(self.policy_model, self.config.lora)
        
        if self.config.training.resume_from_checkpoint:
            print(f"Loading adapter weights from {self.config.training.resume_from_checkpoint}")
            self.policy_model = lora.load_adapter(self.policy_model, self.config.training.resume_from_checkpoint)

        self.ref_model.freeze()
        self.policy_model.from_trainable_parameters = from_trainable_parameters.__get__(self.policy_model, type(self.policy_model))
        
        try:
            self.tokenizer.eos_token_id = self.tokenizer.get_vocab()["<end_of_turn>"]
        except KeyError:
            print("Warning: '<end_of_turn>' token not found. Using default EOS.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def _setup_optimizer_and_params(self):
        self.policy_model.train()
        self.trainable_params = {k: v for k, v in tree_flatten(self.policy_model.parameters()) if "lora" in k}
        if not self.trainable_params:
            raise ValueError("No trainable LoRA parameters found.")
        
        num_params = sum(v.size for _, v in self.trainable_params.items())
        print(f"Found {num_params} LoRA parameters to train.")
        self.optimizer = optim.AdamW(learning_rate=self.config.training.learning_rate)

    def _apply_gradients(self, accumulated_grads: Dict[str, mx.array]):
        # This is the proven-working update logic from your original rl_trainer.py
        if self.config.training.use_lr_scheduler:
            new_lr = cosine_decay_lr(self.step_counter, self.config.training.learning_rate, self.config.training.lr_min, self.config.training.lr_decay_steps)
            self.optimizer.learning_rate = new_lr
        
        grad_values = list(accumulated_grads.values())
        clipped_grad_values, _ = optim.clip_grad_norm(grad_values, self.config.training.grad_clip_norm)
        clipped_grads_dict = dict(zip(accumulated_grads.keys(), clipped_grad_values))
        
        updated_params = self.optimizer.apply_gradients(clipped_grads_dict, self.trainable_params)
        self.policy_model.update(tree_unflatten(list(updated_params.items())))
        
        # This is the original, working line for state management
        self.trainable_params = updated_params
        
        mx.eval(self.policy_model.parameters(), self.optimizer.state)
    
    def train(self):
        train_dataset, self.validation_dataset, test_dataset = prepare_data(self.config)
        print(f"Dataset split: {len(train_dataset)} train, {len(self.validation_dataset)} validation, {len(test_dataset)} test samples.")
        
        win_tracker = self._rehydrate_win_tracker()
        patience = self.config.training.early_stopping_patience
        recent_win_rates, recent_rewards = deque(maxlen=patience), deque(maxlen=patience)
        
        train_steps, train_losses, train_avg_rewards = [], [], []
        eval_steps, eval_win_rates = [], []
        training_game_outcomes: List[GameRecord] = []
        
        print("\n--- Starting Training ---")
        
        data_iterator = iter(itertools.cycle(train_dataset))
        
        start_step = self.step_counter
        step_iterator = range(start_step, self.config.training.iterations + 1)
        pbar = tqdm(step_iterator, desc="Training Steps", total=self.config.training.iterations, initial=start_step)

        for self.step_counter in pbar:
            sample = next(data_iterator)
            # Sampler must be created inside the loop if it has state (it doesn't, but this is safer)
            sampler = make_sampler(
                temp=self.config.rl.sampling_temperature,
            )
            game_rollout = play_wordle_game(
                model=self.policy_model, tokenizer=self.tokenizer, secret_word=sample['secret'],
                system_prompt=self.system_prompt, config=self.config, sampler=sampler,
                initial_history=sample['messages'][1]['content'],
                print_debug=(self.step_counter % self.config.training.log_steps == 0),
                reward_fn=rewards.calculate_total_reward
            )
            
            win_tracker.append(1 if game_rollout.solved else 0)
            rolling_win_rate = (sum(win_tracker) / len(win_tracker)) * 100 if win_tracker else 0.0
            
            avg_loss, accumulated_grads = self._prepare_and_compute_loss(game_rollout)
            
            if accumulated_grads:
                self._apply_gradients(accumulated_grads)
            
            train_record = log_game_result(self.step_counter, avg_loss, game_rollout, 'train')
            training_game_outcomes.append(train_record)
            gc.collect()

            all_rewards = [att.training_reward for att in game_rollout.attempts]
            avg_reward_this_step = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            pbar.set_postfix({"loss": f"{avg_loss:.3f}", "reward": f"{avg_reward_this_step:.2f}", "win%": f"{rolling_win_rate:.1f}"})

            if self.step_counter > 0 and self.step_counter % self.config.evaluation.steps == 0:
                if self._log_and_evaluate(training_game_outcomes, train_steps, train_losses, train_avg_rewards, eval_steps, eval_win_rates, recent_win_rates, recent_rewards, win_tracker):
                    break
                training_game_outcomes.clear()

            if self.step_counter > 0 and self.step_counter % self.config.training.checkpoint_steps == 0:
                checkpoint_file = self.adapter_dir / f"adapter_step_{self.step_counter}.npz"
                lora.save_checkpoint(model=self.policy_model, checkpoint_file=str(checkpoint_file))

        if training_game_outcomes:
            write_metrics_to_file(training_game_outcomes, self.metrics_file_path)
        pbar.close()
        print("\n--- Training Finished ---")
        final_adapter_path = self.adapter_dir / "adapter_final.npz"
        lora.save_checkpoint(model=self.policy_model, checkpoint_file=str(final_adapter_path))
        self.writer.close()
        plot_training_curves(self.run_dir.name, train_steps, train_losses, train_avg_rewards, eval_steps, eval_win_rates, self.plots_dir)
        plot_cumulative_wins(self.metrics_file_path)
        print("Done.")

    def evaluate(self) -> List[GameRecord]:
        self.policy_model.eval()
        eval_game_outcomes = []
        eval_sampler = make_sampler(temp=0.0)
        num_to_select = min(self.config.evaluation.samples, len(self.validation_dataset))
        if num_to_select == 0:
            print("Warning: Evaluation dataset is empty.")
            return []
        
        eval_subset = self.validation_dataset.shuffle(seed=42).select(range(num_to_select))
        for sample in tqdm(eval_subset, desc="Evaluating"):
            game_result = play_wordle_game(
                model=self.policy_model, tokenizer=self.tokenizer, secret_word=sample['secret'],
                system_prompt=self.system_prompt, config=self.config, sampler=eval_sampler,
                initial_history=sample['messages'][1]['content'],
                print_debug=(self.step_counter % 10 == 0), reward_fn=None
            )
            eval_record = log_game_result(self.step_counter, -1.0, game_result, 'eval')
            eval_game_outcomes.append(eval_record)

        self.policy_model.train()
        return eval_game_outcomes

    def _log_and_evaluate(self, training_games, train_steps, train_losses, train_rewards, eval_steps, eval_win_rates, recent_wr, recent_rew, win_tracker):
        if training_games:
            interval_losses = [r.loss_at_step for r in training_games if r.loss_at_step != -1.0]
            avg_loss = np.mean(interval_losses) if interval_losses else -1.0
            interval_rewards = [r.final_reward for r in training_games if r.solved]
            avg_reward = np.mean(interval_rewards) if interval_rewards else 0.0
            if self.config.training.use_early_stopping:
                rolling_win_rate = (sum(win_tracker) / len(win_tracker)) * 100 if win_tracker else 0.0
                if check_early_stopping(recent_wr, recent_rew, rolling_win_rate, avg_reward, self.config.training.early_stopping_patience):
                    return True
            train_steps.append(self.step_counter)
            train_losses.append(avg_loss)
            train_rewards.append(avg_reward)
            log_metrics_to_tensorboard(self.writer, training_games, self.step_counter, 'train')
            write_metrics_to_file(training_games, self.metrics_file_path)

        eval_games = self.evaluate()
        if eval_games:
            wins = sum(1 for r in eval_games if r.solved)
            win_rate = (wins / len(eval_games)) * 100
            win_turns = [r.turns_to_solve for r in eval_games if r.solved]
            avg_turns = np.mean(win_turns) if win_turns else 0.0
            print(f"\n--- Evaluation at Step {self.step_counter:04d} ---")
            print(f"Win Rate: {win_rate:.2f}% | Avg. Turns on Win: {avg_turns:.2f}")
            eval_steps.append(self.step_counter)
            eval_win_rates.append(win_rate)
            log_metrics_to_tensorboard(self.writer, eval_games, self.step_counter, 'eval')
            write_metrics_to_file(eval_games, self.metrics_file_path)
        return False

    def _rehydrate_win_tracker(self) -> deque:
        win_tracker = deque(maxlen=self.config.training.iterations)
        if self.config.training.resume_from_checkpoint:
            truncate_jsonl_log(self.metrics_file_path, self.step_counter - 1)
            return rehydrate_win_tracker(self.metrics_file_path, maxlen=self.config.training.iterations)
        return win_tracker