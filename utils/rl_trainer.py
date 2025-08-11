"""Reinforcement Learning Trainer"""
from typing import List, Any, Dict
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
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from utils import config as cfg
from utils import lora
from utils.rewards_wordle import play_wordle_game
from tensorboardX import SummaryWriter

# ==============================================================================
# ---  HELPER FUNCTIONS (These remain the same) ---
# ==============================================================================
def plot_training_eval(timestamp, train_steps, train_losses, eval_steps, eval_rewards):
    """Generates and saves a plot of training loss and evaluation rewards."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_losses, label="Training Loss")
    plt.plot(eval_steps, eval_rewards, label="Evaluation Reward", marker='o', linestyle='--')
    plt.xlabel("Training Steps")
    plt.ylabel("Value (Loss or Reward)")
    plt.title("Training Loss and Evaluation Reward Curves at " + timestamp)
    plt.legend()
    plt.grid(True)
    plot_filename = f"training_curves_{timestamp}.png"
    plt.savefig(plot_filename)
    print(f"Training curve plot saved to {plot_filename}")

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
# ---  LOSS FUNCTION (Completely replaced with GRPO/DPO loss) ---
# ==============================================================================
def get_log_probs(model: nn.Module, input_ids: mx.array, output_ids: mx.array, pad_token_id: int) -> mx.array:
    full_sequence = mx.concatenate([input_ids, output_ids], axis=1)
    logits = model(full_sequence)
    
    prompt_len = input_ids.shape[1]
    gen_len = output_ids.shape[1]

    output_logits = logits[:, prompt_len - 1:, :]
    
    if output_logits.shape[1] > gen_len:
        output_logits = output_logits[:, :gen_len, :]

    if output_logits.shape[1] != gen_len:
        raise ValueError(f"BUG: Logits shape after slicing ({output_logits.shape}) does not match generation shape ({output_ids.shape}).")

    log_probabilities_stable = nn.log_softmax(output_logits, axis=-1)
    
    chosen_token_log_probs = mx.take_along_axis(
        log_probabilities_stable, output_ids[..., None], axis=-1
    ).squeeze(-1)
    
    mask = (output_ids != pad_token_id).astype(chosen_token_log_probs.dtype)
    masked_log_probs = chosen_token_log_probs * mask
    
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
):
    """
    Calculates the GRPO loss, which is structurally identical to the DPO loss.
    Loss = -log_sigmoid(beta * (log_prob_diff_policy - log_prob_diff_ref))
    """
    # 1. Reconstruct the policy model with the current trainable parameters
    policy_model = policy_model_shell.from_trainable_parameters(trainable_params)

    # 2. Get log probabilities for winner and loser from the policy model
    log_probs_policy_winner = get_log_probs(policy_model, prompt_toks, winner_toks, pad_token_id)
    log_probs_policy_loser = get_log_probs(policy_model, prompt_toks, loser_toks, pad_token_id)

    # 3. Get log probabilities from the frozen reference model (no gradients)
    log_probs_ref_winner = get_log_probs(ref_model, prompt_toks, winner_toks, pad_token_id)
    log_probs_ref_loser = get_log_probs(ref_model, prompt_toks, loser_toks, pad_token_id)

    # 4. Calculate the log-probability differences (policy vs. reference)
    pi_log_ratios = log_probs_policy_winner - log_probs_policy_loser
    ref_log_ratios = log_probs_ref_winner - log_probs_ref_loser

    # 5. The core of the DPO/GRPO loss
    logits = pi_log_ratios - ref_log_ratios
    grpo_loss = -mx.mean(nn.log_sigmoid(config.grpo.beta * logits))

    # 6. Optional: Add a KL divergence penalty to stabilize training
    # This penalizes the policy for moving too far from the reference on the "good" examples
    kl_div = mx.mean(log_probs_policy_winner - log_probs_ref_winner)
    
    loss = grpo_loss + config.grpo.kl_coeff * kl_div
    return loss

# ==============================================================================
# --- EVALUATION FUNCTION (This remains the same) ---
# ==============================================================================
def evaluate(
    model,
    tokenizer,
    dataset,
    config: cfg.TrainerConfig,
    system_prompt: str
) -> Dict[str, Any]:
    """
    Evaluates the model by playing full games of Wordle and tracking win rate.
    """
    model.eval()
    wins = 0
    total_turns_for_wins = 0
    num_samples_processed = 0
    turn_distribution_on_win = defaultdict(int)
    eval_sampler = make_sampler(temp=0.0)
    num_to_select = min(config.evaluation.samples, len(dataset))
    if num_to_select == 0:
        print("Warning: Evaluation dataset is empty. Skipping evaluation.")
        model.train()
        return {'win_rate': 0.0, 'avg_turns_on_win': 0.0, 'turn_distribution_on_win': {}}
    eval_subset = dataset.shuffle(seed=42).select(range(num_to_select))
    for sample in tqdm(eval_subset, desc="Evaluating"):
        secret_word = sample['secret']
        game_result = play_wordle_game(
            model=model,
            tokenizer=tokenizer,
            secret_word=secret_word,
            system_prompt=system_prompt,
            config=config,
            sampler=eval_sampler
        )
        num_samples_processed += 1
        if game_result.solved:
            wins += 1
            num_turns = len(set(att.prompt_string for att in game_result.attempts))
            total_turns_for_wins += num_turns
            turn_distribution_on_win[num_turns] += 1
    model.train()
    win_rate = (wins / num_samples_processed) if num_samples_processed > 0 else 0.0
    avg_turns_on_win = (total_turns_for_wins / wins) if wins > 0 else 0.0
    return {
        'win_rate': win_rate,
        'avg_turns_on_win': avg_turns_on_win,
        'turn_distribution_on_win': dict(sorted(turn_distribution_on_win.items()))
    }

def load_wordle_trajectories_from_jsonl(dataset_path: str) -> Dataset:
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

# ==============================================================================
# --- MAIN TRAINING ---
# ==============================================================================
def train(config: cfg.TrainerConfig, system_prompt: str):
    """Main training loop for the GRPO-based Wordle solver."""
    print(f"MLX is using default device: {mx.default_device()}")
    print(f"Loading base model: {config.model.name}")
    print(f"Resume from checkpoint: {config.training.resume_from_checkpoint}")

    dataset_path = "./data/sft_wordle_cot_data.jsonl"
    adapter_path = Path(config.training.save_adapter_to)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/rl_wordle_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    win_tracker = deque(maxlen=config.training.iterations)

    # In GRPO/DPO, the 'reference model' is a frozen copy of the original SFT model.
    # It does NOT get updated during training.
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
    optimizer = optim.AdamW(learning_rate=config.training.learning_rate)

    try:
        tokenizer.eos_token_id = tokenizer.get_vocab()["<end_of_turn>"]
    except KeyError:
        print("Warning: '<end_of_turn>' token not found. Using default EOS.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    dataset = load_wordle_trajectories_from_jsonl(dataset_path)
    shuffled_dataset = dataset.shuffle(seed=42)
    
    train_percentage = 0.95
    num_train_samples = int(len(shuffled_dataset) * train_percentage)
    train_dataset = shuffled_dataset.select(range(num_train_samples))
    test_dataset = shuffled_dataset.select(range(num_train_samples, len(shuffled_dataset)))

    loss_and_grad_fn = mx.value_and_grad(grpo_loss_and_grad, argnums=0)
    train_steps, train_losses = [], []
    eval_steps, eval_rewards = [], []

    print("\nStarting GRPO training...")
    step_counter = 0
    sampler = make_sampler(temp=config.rl.sampling_temperature)
    data_iterator = iter(itertools.cycle(train_dataset))
    pbar = tqdm(total=config.training.iterations, desc="GRPO Training Steps")

    while step_counter < config.training.iterations:
        sample = next(data_iterator)
        game_rollout = play_wordle_game(
            model=policy_model, tokenizer=tokenizer, secret_word=sample['secret'],
            system_prompt=system_prompt, config=config, sampler=sampler, initial_history=sample['messages']
        )
        
        win_tracker.append(1 if game_rollout.solved else 0)
        rolling_win_rate = (sum(win_tracker) / len(win_tracker)) * 100 if win_tracker else 0.0

        if not game_rollout.attempts:
            pbar.update(1)
            continue

        grouped_attempts = defaultdict(list)
        for attempt in game_rollout.attempts:
            grouped_attempts[attempt.prompt_string].append(attempt)

        accumulated_grads = {k: mx.zeros_like(v) for k, v in trainable_params.items()}
        total_loss = 0.0
        num_micro_batches = 0

        for _, attempts_for_prompt in grouped_attempts.items():
            if len(attempts_for_prompt) < 2:
                continue # Need at least one winner and one loser for a comparison

            winner = max(attempts_for_prompt, key=lambda att: att.reward)
            
            # Create all (winner, loser) pairs for this prompt
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
        
        if num_micro_batches > 0:
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

        gc.collect()

        step_counter += 1
        pbar.update(1)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "win rate": f"{rolling_win_rate:.1f}%"})

        # --- LOGGING, EVALUATION, AND CHECKPOINTING ---
        if step_counter > 0 and step_counter % config.training.log_steps == 0:
            print(f"\nStep {step_counter:04d} | Avg Train Loss: {avg_loss:.4f}")
            train_steps.append(step_counter)
            train_losses.append(avg_loss)
            writer.add_scalar('Loss/train', avg_loss, step_counter)

        if step_counter > 0 and step_counter % config.evaluation.steps == 0:
            eval_metrics = evaluate(policy_model, tokenizer, test_dataset, config, system_prompt)
            win_rate = eval_metrics['win_rate'] * 100
            print(f"\n--- Evaluation at Step {step_counter:04d} ---")
            print(f"Win Rate: {win_rate:.2f}% | Avg. Turns on Win: {eval_metrics['avg_turns_on_win']:.2f}")
            eval_steps.append(step_counter)
            eval_rewards.append(eval_metrics['win_rate'])
            writer.add_scalar('Evaluation/win_rate', win_rate, step_counter)

        # NB: There is no reference model update in GRPO/DPO
        
        if step_counter > 0 and step_counter % config.training.checkpoint_steps == 0:
            lora.save_checkpoint(model=policy_model,  save_dir=str(adapter_path), checkpoint_file_name="grpo_lora_wordle", 
                                                   step=str(step_counter), timestamp=timestamp)

    pbar.close()
    print("\n--- Training Finished ---")
    lora.save_checkpoint(policy_model, adapter_path.stem, "final", timestamp)
    writer.close()
    plot_training_eval(timestamp, train_steps, train_losses, eval_steps, eval_rewards)
    print("Done.")