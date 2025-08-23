"""Reinforcement Learning Trainer"""
from typing import List
import itertools
import gc
from collections import defaultdict, deque
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json
from datasets import Dataset
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from utils import config as cfg
from ml import lora
from tensorboardX import SummaryWriter
from functools import partial
import numpy as np
import math
import re
from wordle.game import GameRecord, play_wordle_game, parse_guess
from wordle import rewards
from utils.logging import log_game_result, write_metrics_to_file, log_metrics_to_tensorboard, truncate_jsonl_log, plot_training_curves, plot_cumulative_wins


# ==============================================================================
# ---  HELPER FUNCTIONS ---
# ==============================================================================
def from_trainable_parameters(model_instance: nn.Module, trainable_params: dict):
    model_instance.update(tree_unflatten(list(trainable_params.items())))
    return model_instance

def get_named_parameters_flat(model_params: dict, prefix: str = ''):
    flat_params = []
    for name, param in model_params.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(param, dict):
            flat_params.extend(get_named_parameters_flat(param, prefix=full_name))
        elif isinstance(param, list):
            for i, sub_param in enumerate(param):
                flat_params.extend(get_named_parameters_flat(sub_param, prefix=f"{full_name}.{i}"))
        else:
            flat_params.append((full_name, param))
    return flat_params

def pad_sequences(token_lists: List[List[int]], pad_value: int) -> mx.array:
    if not token_lists:
        return mx.array([], dtype=mx.int32)
    max_len = max(len(tokens) for tokens in token_lists)
    padded_lists = [tokens + [pad_value] * (max_len - len(tokens)) for tokens in token_lists]
    return mx.array(padded_lists)

def is_nan_or_inf(x):
    return mx.isnan(x).any().item() or mx.isinf(x).any().item()
# ==============================================================================
# ---  LOSS FUNCTION ---
# ==============================================================================
def cosine_decay_lr(step: int, initial_lr: float, min_lr: float, decay_steps: int) -> float:
    """
    Calculates the learning rate at a given step using a cosine decay schedule.
    """
    if step >= decay_steps:
        return min_lr
    
    # Cosine annealing formula
    decay_ratio = step / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (initial_lr - min_lr)

def get_log_probs(model: nn.Module, input_ids: mx.array, output_ids: mx.array, pad_token_id: int) -> mx.array:
    full_sequence = mx.concatenate([input_ids, output_ids], axis=1)
    logits = model(full_sequence)
    
    prompt_len = input_ids.shape[1]
    gen_len = output_ids.shape[1]

    # Slice logits to only include those corresponding to the generated tokens
    output_logits = logits[:, prompt_len - 1:, :]
    
    if output_logits.shape[1] > gen_len:
        output_logits = output_logits[:, :gen_len, :]

    if output_logits.shape[1] != gen_len:
        raise ValueError(f"BUG: Logits shape after slicing ({output_logits.shape}) does not match generation shape ({output_ids.shape}).")

    # (batch_size, gen_len, vocab_size)
    log_probabilities_stable = nn.log_softmax(output_logits, axis=-1)

    # pluck out the log-probabilities of the actually chosen tokens in the output_ids
    # (batch_size, gen_len)
    # `chosen_token_log_probs` = `[ log(P(T1)), log(P(T2 | T1)), log(P(T3 | T1, T2)) ]`
    chosen_token_log_probs = mx.take_along_axis(
        log_probabilities_stable, output_ids[..., None], axis=-1
    ).squeeze(-1)
    
    # Mask out padding tokens
    mask = (output_ids != pad_token_id).astype(chosen_token_log_probs.dtype)
    masked_log_probs = chosen_token_log_probs * mask
    # log of all the generated sequence tokens (sum over sequence length)
    total_log_prob = mx.sum(masked_log_probs, axis=-1)
    
    return total_log_prob

def grpo_loss_and_grad(
    trainable_params: dict,
    policy_model_shell: nn.Module,
    ref_model: nn.Module,
    winner_toks: mx.array,
    loser_toks: mx.array,
    prompt_toks: mx.array,
    config: cfg.TrainerConfig,
    pad_token_id: int,
) -> mx.array:
    """
    Calculates the GRPO loss.
    Loss = -log_sigmoid(beta * (log_prob_diff_policy - log_prob_diff_ref))
    
    Note: this fn will be called with JAX-style value_and_grad and will 
    return both loss and grads, Tuple[mx.array, Dict].
    This is a dictionary containing the gradients of the loss with respect
    to the first argument (argnums=0) of the original function, which is trainable_params.
    The grads dictionary will have the same structure (keys and shapes) as our trainable_params dictionary.

    Args:
        trainable_params: Current LoRA trainable parameters of the policy model.
        policy_model_shell: The policy model class (with LoRA) to be reconstructed.
        ref_model: The frozen reference model (without LoRA).
        winner_toks: Token IDs of the winning responses (batch_size, seq_len).
        loser_toks: Token IDs of the losing responses (batch_size, seq_len).
        prompt_toks: Token IDs of the prompts (batch_size, prompt_len).
        config: Trainer configuration with GRPO hyperparameters.
        pad_token_id: Token ID used for padding.

    Returns:
        loss: The computed GRPO loss as a scalar mx.array.
    """
    # 1. Reconstruct the policy model with the current LoRA trainable parameters
    policy_model = policy_model_shell.from_trainable_parameters(trainable_params)

    # 2. Get log probabilities for winner and loser from the policy model
    log_probs_policy_winner = get_log_probs(policy_model, prompt_toks, winner_toks, pad_token_id)
    log_probs_policy_loser = get_log_probs(policy_model, prompt_toks, loser_toks, pad_token_id)

    # 3. Get log probabilities from the frozen reference model (no LoRA)
    log_probs_ref_winner = get_log_probs(ref_model, prompt_toks, winner_toks, pad_token_id)
    log_probs_ref_loser = get_log_probs(ref_model, prompt_toks, loser_toks, pad_token_id)

    # 4. Calculate the log-probability differences (policy vs. reference)
    pi_log_ratios = log_probs_policy_winner - log_probs_policy_loser
    ref_log_ratios = log_probs_ref_winner - log_probs_ref_loser

    # 5. The core of the GRPO loss
    logits = pi_log_ratios - ref_log_ratios
    grpo_loss = -mx.mean(nn.log_sigmoid(config.grpo.beta * logits))

    # 6. Add a KL divergence penalty to stabilize training
    # This penalizes the policy for moving too far from the reference on the "good" examples
    # TODO : consider removing this as mixtral removed it in https://arxiv.org/pdf/2506.10910
    kl_div = mx.mean(log_probs_ref_winner - log_probs_policy_winner)
    kl_penalty = mx.maximum(0, kl_div)

    loss = grpo_loss + config.grpo.kl_coeff * kl_penalty
    return loss

# ==============================================================================
# --- EVALUATION FUNCTION ---
# ==============================================================================
def evaluate(
    model,
    tokenizer,
    dataset,
    config: cfg.TrainerConfig,
    system_prompt: str,
    current_step: int
) -> List[GameRecord]:
    """
    Evaluates the model and returns a list of detailed GameRecord objects.
    """
    # go into eval mode
    model.eval()
    eval_game_outcomes: List[GameRecord] = []
    # make a sampler with temperature 0 for deterministic evaluation
    eval_sampler = make_sampler(temp=0.0)
    num_to_select = min(config.evaluation.samples, len(dataset))
    if num_to_select == 0:
        print("Warning: Evaluation dataset is empty. Skipping evaluation.")
        model.train()
        return []
    eval_subset = dataset.shuffle(seed=42).select(range(num_to_select))
    for sample in tqdm(eval_subset, desc="Evaluating"):
        secret_word = sample['secret']
        game_result = play_wordle_game(
            model=model,
            tokenizer=tokenizer,
            secret_word=secret_word,
            system_prompt=system_prompt,
            config=config,
            sampler=eval_sampler,
            initial_history=sample['messages'][1]['content'],
            print_debug=(current_step % 10 == 0),
            reward_fn=None
        )
        eval_record = log_game_result(current_step, -1.0, game_result, 'eval')
        eval_game_outcomes.append(eval_record)

    model.train()
    return eval_game_outcomes
# ==============================================================================
# --- DATA LOADING ---
# ==============================================================================
def load_wordle_trajectories_from_jsonl(dataset_path: str) -> Dataset:
    """Loads Wordle game trajectories from a JSONL file into a HuggingFace Dataset."""
    print(f"Loading game trajectories from: {dataset_path}")
    game_trajectories = []
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data_point = json.loads(line)
                    data_content = data_point.get("data", {})
                    secret_word = data_content.get("secret")
                    messages = data_content.get("messages")
                    
                    if secret_word and messages:
                        game_trajectories.append({
                            "secret": secret_word.upper(),
                            "messages": messages
                        })
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at '{dataset_path}'")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse JSON from '{dataset_path}'. Error: {e}")
        raise

    if not game_trajectories:
        raise ValueError("No game trajectories were loaded from the dataset.")

    dataset = Dataset.from_list(game_trajectories)
    print(f"Successfully loaded {len(dataset)} game trajectories.")
    return dataset


def is_playable_trajectory(example, max_trials: int):
    """
    Checks if a game trajectory is playable.

    A trajectory is NOT playable if:
    1. It has already been solved.
    2. It has already reached or exceeded the maximum number of trials.
    """
    secret_word = example['secret'].upper()
    assistant_messages = [m for m in example['messages'] if m.get("role") == "assistant"]

    # Check 2: Has the game already used all its turns?
    if len(assistant_messages) >= max_trials:
        return False # This game is already over (likely a loss)

    # Check 1: Has the game already been solved?
    for message in assistant_messages:
        guess = parse_guess(message.get("content", ""))
        if guess and guess == secret_word:
            return False # This game was already solved
            
    return True # This game is unsolved and has turns remaining

# ==============================================================================
# --- MAIN TRAINING ---
# ==============================================================================
def train(config: cfg.TrainerConfig, system_prompt: str):
    """Main training loop for the GRPO-based Wordle solver."""
    print(f"MLX is using default device: {mx.default_device()}")
    print(f"Loading base model: {config.model.name}")

    step_counter = 1
    if config.training.resume_from_checkpoint:
        checkpoint_path = Path(config.training.resume_from_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found at: {checkpoint_path}")

        run_dir = checkpoint_path.parent.parent
        run_name = run_dir.name
        timestamp = run_name.split('_')[0]
        
        match = re.search(r'_(\d+)\.npz$', checkpoint_path.name)
        if match:
            # The checkpoint was saved *after* the completed_step was done.
            # So we start the next iteration.
            step_counter = int(match.group(1)) + 1
            print(f"Resuming from timestamp: {timestamp} at step {step_counter}")
        else:
            raise ValueError(f"Could not parse step number from checkpoint name: {checkpoint_path.name}")

        print(f"Resuming run from directory: {run_dir}")
        print(f"Resuming from timestamp: {timestamp} at step {step_counter}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name_short = config.model.name.split('/')[-1]
        run_name = f"{timestamp}_{model_name_short}_rank{config.lora.rank}"
        run_dir = Path("experiments") / run_name
        
        print(f"Starting new run. All artifacts will be saved in: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = f"{run_dir}/{config.training.config_file}"
        cfg.save_config(config, config_path)

    # --- 2. DEFINE ALL PATHS ---
    adapter_dir = run_dir / "adapters"
    tensorboard_dir = run_dir / "tensorboard"
    plots_dir = run_dir / "plots"
    metrics_file_path = run_dir / "training_metrics.jsonl"

    for d in [adapter_dir, tensorboard_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- 3. SETUP LOGGING ---
    print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    writer = SummaryWriter(str(tensorboard_dir))

    print(f"Detailed game metrics will be saved to: {metrics_file_path}")
    if config.training.resume_from_checkpoint and step_counter > 0:
        print(f"Truncating logs to step {step_counter}...")
        truncate_jsonl_log(metrics_file_path, step_counter)

    dataset_path = config.training.data_path
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found at '{dataset_path}'")
    print(f"Loading dataset from: {dataset_path}")
    win_tracker = deque(maxlen=config.training.iterations)

    # In GRPO, the 'reference model' is a frozen copy of the original model.
    # It does NOT get updated during training.
    # TODO: consider updating the reference model periodically every x steps.
    print("Creating policy and reference models...")
    policy_model, tokenizer = load(config.model.name)
    ref_model, _ = load(config.model.name)

    # Apply LoRA layers to the policy model, which will be trained.
    lora_config = {"rank": config.lora.rank, "alpha": config.lora.alpha, "dropout": config.lora.dropout}
    policy_model = lora.apply_lora_to_model(policy_model, lora_config, config.lora.layers_to_tune)
    
    if config.training.resume_from_checkpoint:
        print(f"\n--- Resuming training, loading adapter: {config.training.resume_from_checkpoint} ---")
        policy_model = lora.load_adapter(policy_model, config.training.resume_from_checkpoint)

    # The reference model remains frozen without LoRA adapters.
    ref_model.freeze()
    policy_model.from_trainable_parameters = from_trainable_parameters.__get__(policy_model, type(policy_model))

    trainable_params = { k: v for k, v in tree_flatten(policy_model.parameters()) if "lora" in k }
    if not trainable_params:
        raise ValueError("No trainable LoRA parameters found.")
    print(f"Found {len(trainable_params)} LoRA parameters to train.")

    policy_model.train()
    # TODO: consider using LoRA Rite from https://github.com/gkevinyen5418/LoRA-RITE
    # needs to be adapted to MLX from pyTorch
    optimizer = optim.AdamW(learning_rate=config.training.learning_rate)

    try:
        tokenizer.eos_token_id = tokenizer.get_vocab()["<end_of_turn>"]
    except KeyError:
        print("Warning: '<end_of_turn>' token not found. Using default EOS.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    dataset = load_wordle_trajectories_from_jsonl(dataset_path)
    print(f"Original dataset size: {len(dataset)}")

    # Create a filter function that knows about max_trials from your config
    max_trials_for_filter = config.rl.max_trials
    playable_filter = partial(is_playable_trajectory, max_trials=max_trials_for_filter)

    # Apply filter
    playable_dataset = dataset.filter(playable_filter)
    print(f"Filtered dataset size (playable games only): {len(playable_dataset)}")

    # Now, use the fully cleaned dataset for training
    shuffled_dataset = playable_dataset.shuffle(seed=42)
    
    train_percentage = 0.85 # Use a 85/15 split

    num_train_samples = int(len(shuffled_dataset) * train_percentage)
    train_dataset = shuffled_dataset.select(range(num_train_samples))
    test_dataset = shuffled_dataset.select(range(num_train_samples, len(shuffled_dataset)))
    print(f"Dataset split: {len(train_dataset)} training samples, {len(test_dataset)} evaluation samples.")
    # This fn will be called with JAX-style value_and_grad and will 
    # return both loss and grads, Tuple[mx.array, Dict].
    # This is a dictionary containing the gradients of the loss with respect
    # to the first argument (argnums=0) of the original function, which is trainable_params.
    # The grads dictionary will have the same structure (keys and shapes) as our trainable_params dictionary (LoRA).
    loss_and_grad_fn = mx.value_and_grad(grpo_loss_and_grad, argnums=0)
    train_steps, train_losses, train_avg_rewards = [], [], []
    eval_steps, eval_win_rates = [], []
    training_game_outcomes: List[GameRecord] = []

    print("\nStarting GRPO training...")
    sampler = make_sampler(temp=config.rl.sampling_temperature)
    data_iterator = iter(itertools.cycle(train_dataset))

    start_step = step_counter
    step_iterator = range(start_step, config.training.iterations + 1)
    pbar = tqdm(step_iterator,
                desc="GRPO Training Steps",
                total=config.training.iterations,
                initial=start_step)

    for step_counter in pbar:
        sample = next(data_iterator)
        game_rollout = play_wordle_game(
            model=policy_model, tokenizer=tokenizer, secret_word=sample['secret'],
            system_prompt=system_prompt, config=config, sampler=sampler, initial_history=sample['messages'][1]['content'],
            print_debug=(step_counter % config.training.log_steps == 0),
            reward_fn=rewards.calculate_total_reward
        )
        
        win_tracker.append(1 if game_rollout.solved else 0)
        rolling_win_rate = (sum(win_tracker) / len(win_tracker)) * 100 if win_tracker else 0.0
        if not game_rollout.attempts:
            print("\n⚠️ Warning: Game rollout produced no attempts. This step will have no loss/gradient update. ⚠️")

        all_rewards = [att.training_reward for att in game_rollout.attempts]
        avg_reward_this_step = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        # Group attempts by their prompt string to prompt
        grouped_attempts = defaultdict(list)
        for attempt in game_rollout.attempts:
            grouped_attempts[attempt.prompt_string].append(attempt)

        accumulated_grads = {k: mx.zeros_like(v) for k, v in trainable_params.items()}
        total_loss = 0.0
        num_micro_batches = 0

        # prepare all (winner, loser) pairs and compute gradients
        for _, attempts_for_prompt in grouped_attempts.items():
            # Need at least one winner and one loser for a comparison
            if len(attempts_for_prompt) < 2:
                continue

            # Identify the winner (highest reward) for this prompt
            winner = max(attempts_for_prompt, key=lambda att: att.training_reward)
            
            # Create all (winner, loser) pairs of generated responses for this prompt
            winner_toks_list, loser_toks_list, prompt_toks_list = [], [], []
            for loser in attempts_for_prompt:
                if loser is not winner:
                    winner_toks_list.append(winner.response_tokens)
                    loser_toks_list.append(loser.response_tokens)
                    prompt_toks_list.append(winner.prompt_tokens)
            
            if not loser_toks_list:
                continue

            # Pad all sequences to the same length for batching
            winner_toks_padded = pad_sequences(winner_toks_list, pad_id)
            loser_toks_padded = pad_sequences(loser_toks_list, pad_id)
            prompt_toks_padded = pad_sequences(prompt_toks_list, pad_id)

            loss, micro_grads = loss_and_grad_fn(
                trainable_params,
                policy_model,
                ref_model,
                winner_toks_padded,
                loser_toks_padded,
                prompt_toks_padded,
                config,
                pad_id
            )
            mx.eval(loss, micro_grads)
            
            if is_nan_or_inf(loss):
                print("\n⚠️ DETECTED NaN/Inf IN GRADIENTS. SKIPPING STEP. ⚠️")
                continue

            for key, grad_val in micro_grads.items():
                accumulated_grads[key] += grad_val

            total_loss += loss.item()
            num_micro_batches += 1
        
        # After processing all prompts in this step, average gradients and apply update when we have valid micro-batches 
        if num_micro_batches > 0:

            ## LR Decay
            if config.training.use_lr_scheduler:
                # Calculate the new learning rate for the current step
                new_lr = cosine_decay_lr(
                                step=step_counter,
                                initial_lr=config.training.learning_rate,
                                min_lr=config.training.lr_min,
                                decay_steps=config.training.lr_decay_steps
                            )
                # Apply the new learning rate to the optimizer
                optimizer.learning_rate = new_lr
        
            # Average the accumulated gradients and loss
            avg_grads = {k: v / num_micro_batches for k, v in accumulated_grads.items()}
            avg_loss = total_loss / num_micro_batches

            grad_values = list(avg_grads.values())
            clipped_grad_values, _ = optim.clip_grad_norm(grad_values, 1.0)
            clipped_grads_dict = dict(zip(avg_grads.keys(), clipped_grad_values))
            
            updated_params = optimizer.apply_gradients(clipped_grads_dict, trainable_params)
            policy_model.update(tree_unflatten(list(updated_params.items())))
            trainable_params = updated_params
            mx.eval(policy_model.parameters(), optimizer.state)
        else:
            avg_loss = -1.0

        train_record = log_game_result(step_counter, avg_loss, game_rollout, 'train')
        training_game_outcomes.append(train_record)
        gc.collect()

        pbar.set_postfix({
            "loss": f"{avg_loss:.3f}", 
            "reward": f"{avg_reward_this_step:.2f}",
            "win%": f"{rolling_win_rate:.1f}\n"
        })

        if step_counter > 0 and step_counter % config.evaluation.steps == 0:
            # We average the metrics from all training games played since the last evaluation.
            # This creates a much smoother and more meaningful trendline.
           
            # Calculate and store metrics for the plot form training
            if training_game_outcomes:
                interval_losses = [r.loss_at_step for r in training_game_outcomes if r.loss_at_step != -1.0]
                avg_interval_loss = np.mean(interval_losses) if interval_losses else -1.0
                
                interval_win_rewards = [r.final_reward for r in training_game_outcomes if r.solved]
                avg_interval_reward = np.mean(interval_win_rewards) if interval_win_rewards else 0.0
                
                train_steps.append(step_counter)
                train_losses.append(avg_interval_loss)
                train_avg_rewards.append(avg_interval_reward)

            # Log training data to Tensorboard and files
            print(f"\n--- Logging metrics for training steps up to {step_counter} ---")
            log_metrics_to_tensorboard(writer, training_game_outcomes, step_counter, 'train')
            write_metrics_to_file(training_game_outcomes, metrics_file_path)
            training_game_outcomes.clear()

            # Evaluate the model on the test dataset
            eval_game_outcomes = evaluate(
                policy_model, tokenizer, test_dataset, config, system_prompt, step_counter)
            if eval_game_outcomes:
                eval_wins = sum(1 for r in eval_game_outcomes if r.solved)
                eval_total = len(eval_game_outcomes)
                win_rate = (eval_wins / eval_total) * 100 if eval_total > 0 else 0.0
                
                win_turns = [r.turns_to_solve for r in eval_game_outcomes if r.solved]
                avg_turns_on_win = np.mean(win_turns) if win_turns else 0.0

                print(f"\n--- Evaluation at Step {step_counter:04d} ---")
                print(f"Win Rate: {win_rate:.2f}% | Avg. Turns on Win: {avg_turns_on_win:.2f}")
                
                # Store the results for the final plot
                eval_steps.append(step_counter)
                eval_win_rates.append(win_rate)
            log_metrics_to_tensorboard(writer, eval_game_outcomes, step_counter, 'eval')
            write_metrics_to_file(eval_game_outcomes, metrics_file_path)

        # Checkpointing
        if step_counter > 0 and step_counter % config.training.checkpoint_steps == 0:
            checkpoint_file = adapter_dir / f"adapter_step_{step_counter}.npz"
            lora.save_checkpoint(model=policy_model,  checkpoint_file=str(checkpoint_file))

    if training_game_outcomes:
        print(f"\nWriting final {len(training_game_outcomes)} training metrics to file...")
        write_metrics_to_file(training_game_outcomes, metrics_file_path)
    pbar.close()
    print("\n--- Training Finished ---")
    final_adapter_path = adapter_dir / "adapter_final.npz"
    lora.save_checkpoint(model=policy_model, checkpoint_file=str(final_adapter_path))
    writer.close()
    plot_training_curves(
        timestamp, 
        train_steps, 
        train_losses, 
        train_avg_rewards, 
        eval_steps, 
        eval_win_rates,
        plots_dir
    )
    plot_cumulative_wins(metrics_file_path)
    print("Done.")