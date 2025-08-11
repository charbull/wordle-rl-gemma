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
# ---  HELPER & REWARD FUNCTIONS ---
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
    """
    A helper function to update a model with a dictionary of its trainable parameters.
    """
    # This correctly uses tree_unflatten for a dictionary.
    model_instance.update(tree_unflatten(list(trainable_params.items())))
    return model_instance

def get_named_parameters_flat(model_params: dict, prefix: str = ''):
    """
    A robust helper function to recursively flatten a nested structure of parameters
    and return a list of (name, value) pairs.
    """
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
    """
    Pads a list of token lists to the same length and returns an mx.array.
    """
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
# In rl_trainer.py

import mlx.nn as nn
import mlx.core as mx

def get_log_probs(model: nn.Module, input_ids: mx.array, output_ids: mx.array, pad_token_id: int) -> mx.array:
    """
    Calculates the total log probability of a model generating `output_ids` given `input_ids`.
    
    This function uses the fundamental definition: log(softmax) followed by gathering
    the probabilities of the chosen tokens. It is numerically stabilized by using
    the log_softmax function.

    Args:
        model: The language model (e.g., the policy model).
        input_ids: A batch of tokenized prompts, shape (batch_size, prompt_len).
        output_ids: A batch of tokenized generations, shape (batch_size, gen_len).
        pad_token_id: The ID of the padding token to be ignored in the calculation.

    Returns:
        A 1D mx.array of shape (batch_size,) containing the total log probability
        for each sequence in the batch.
    """
    full_sequence = mx.concatenate([input_ids, output_ids], axis=1)
    logits = model(full_sequence)
    
    prompt_len = input_ids.shape[1]
    gen_len = output_ids.shape[1]

    # Slice to get only the logits for the generated tokens
    output_logits = logits[:, prompt_len - 1:, :]
    
    # Truncate if the model produces one extra logit
    if output_logits.shape[1] > gen_len:
        output_logits = output_logits[:, :gen_len, :]

    if output_logits.shape[1] != gen_len:
        raise ValueError(f"BUG: Logits shape after slicing ({output_logits.shape}) does not match generation shape ({output_ids.shape}).")

    # Use the stable log_softmax function to get log probabilities for all possible tokens.
    log_probabilities_stable = nn.log_softmax(output_logits, axis=-1)
    
    # Gather the log probabilities of only the specific tokens that were actually generated.
    chosen_token_log_probs = mx.take_along_axis(
        log_probabilities_stable, output_ids[..., None], axis=-1
    ).squeeze(-1)
    
    # Create a mask to ignore padding tokens
    mask = (output_ids != pad_token_id).astype(chosen_token_log_probs.dtype)
    masked_log_probs = chosen_token_log_probs * mask
    
    # The result is already a log probability, so we just sum it.
    total_log_prob = mx.sum(masked_log_probs, axis=-1)
    
    return total_log_prob

def grpo_loss_and_grad(
    trainable_params: dict,
    model_shell: nn.Module,
    prompt_toks_padded: mx.array,
    gen_toks_padded: mx.array,
    advantages: mx.array,
    log_probs_ref: mx.array,
    config: cfg.TrainerConfig,
    pad_token_id: int,
):
    policy_model = model_shell.from_trainable_parameters(trainable_params)
    log_probs_current = get_log_probs(policy_model, prompt_toks_padded, gen_toks_padded, pad_token_id)
    log_ratios = log_probs_current - log_probs_ref
    # clip log ratios to avoid extreme values
    LOG_RATIO_CLIP = 4.0 
    log_ratios = mx.clip(log_ratios, a_min=-LOG_RATIO_CLIP, a_max=LOG_RATIO_CLIP)
    ratios = mx.exp(log_ratios)
    unclipped_objective = ratios * advantages
    clipped_objective = mx.clip(ratios, 1 - config.ppo.clip_epsilon, 1 + config.ppo.clip_epsilon) * advantages
    ppo_objective = mx.minimum(unclipped_objective, clipped_objective)
    ppo_loss = -mx.mean(ppo_objective)
    kl_div = mx.mean(log_ratios)
    loss = ppo_loss + config.ppo.kl_coeff * kl_div
    return loss

# ==============================================================================
# --- EVALUATION FUNCTION ---
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
    """
    Loads Wordle game trajectories from a JSON Lines file.

    Each line is a JSON object containing a 'secret' word and a 'messages'
    list representing a partial or full game history.

    Args:
        dataset_path: The path to the .jsonl file.

    Returns:
        A Dataset object with 'secret' and 'messages' columns.
    """
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
        print("Cannot proceed without a dataset for this training type.")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse JSON from '{dataset_path}'. Error: {e}")
        raise

    if not game_trajectories:
        raise ValueError("No game trajectories were loaded from the dataset.")

    # Create a Dataset object from the list of dictionaries
    dataset = Dataset.from_list(game_trajectories)
    print(f"Successfully loaded {len(dataset)} game trajectories.")
    return dataset

# ==============================================================================
# --- MAIN TRAINING  ---
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
    quantization_config = {"group_size": 64, "bits": 4}
    model_config={"quantize": quantization_config}
    win_tracker = deque(maxlen=config.training.iterations)

    if config.training.resume_from_checkpoint:
        print(f"\n--- Resuming training from: {config.training.resume_from_checkpoint} ---")
        policy_model, tokenizer = load(
            path_or_hf_repo=config.model.name,
            adapter_path=config.training.resume_from_checkpoint
        )
    else:
        print(f"\n--- Starting training from scratch. ---")
        policy_model, tokenizer = load(config.model.name, model_config)
        print("Applying fresh LoRA layers to the model...")
        lora_config = {"rank": config.lora.rank, "alpha": config.lora.alpha, "dropout": config.lora.dropout}
        policy_model = lora.apply_lora_to_model(policy_model, lora_config, config.lora.layers_to_tune)

    policy_model.from_trainable_parameters = from_trainable_parameters.__get__(policy_model, type(policy_model))

    all_params_flat = tree_flatten(policy_model.parameters())
    trainable_params = {
        k: v for k, v in all_params_flat if "lora" in k
    }
#     trainable_params = {
#     k: v for k, v in tree_flatten(policy_model.parameters()) if "lora" in k
# }
    if not trainable_params:
        raise ValueError("No trainable LoRA parameters found.")
    print(f"Found {len(trainable_params)} LoRA parameters to train.")

    policy_model.freeze()
    mx.eval(policy_model.parameters())

    print("Creating reference model...")
    ref_model, _ = load(config.model.name, model_config)
    lora_config = {"rank": config.lora.rank, "alpha": config.lora.alpha, "dropout": config.lora.dropout}
    ref_model = lora.apply_lora_to_model(ref_model, lora_config, config.lora.layers_to_tune)
    ref_model.freeze()
    ref_model.update(policy_model.parameters())
    print("Reference model created and synchronized.")

    policy_model.train()
    optimizer = optim.AdamW(learning_rate=config.training.learning_rate)

    try:
        tokenizer.eos_token_id = tokenizer.get_vocab()["<end_of_turn>"]
    except KeyError:
        print("Warning: '<end_of_turn>' token not found. Using default EOS.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id


    print("Loading dataset: ", dataset_path)
    # Call the new helper function to load the data from your custom file.
    dataset = load_wordle_trajectories_from_jsonl(dataset_path)

    # The rest of the data processing remains the same
    shuffled_dataset = dataset.shuffle(seed=42)
    
    # Split the data into training and testing sets
    train_percentage = 0.95
    if len(shuffled_dataset) > 1:
        num_train_samples = int(len(shuffled_dataset) * train_percentage)
        train_dataset = shuffled_dataset.select(range(num_train_samples))
        test_dataset = shuffled_dataset.select(range(num_train_samples, len(shuffled_dataset)))
    else: # Handle cases with very small datasets
        train_dataset = shuffled_dataset
        test_dataset = shuffled_dataset


    trainable_params = {
        k: v for k, v in get_named_parameters_flat(policy_model.parameters()) if "lora" in k
    }

    loss_and_grad_fn = mx.value_and_grad(grpo_loss_and_grad, argnums=0)
    train_steps, train_losses = [], []
    eval_steps, eval_rewards = [], []

    print("\nStarting GRPO training...")
    step_counter = 0
    initial_temp = config.rl.sampling_temperature
    min_temp = 0.4
    sampler = make_sampler(temp=initial_temp)
    data_iterator = iter(itertools.cycle(train_dataset))
    pbar = tqdm(total=config.training.iterations, desc="GRPO Training Steps")

    while step_counter < config.training.iterations:
        sample = next(data_iterator)
        secret_word = sample['secret']
        message_history = sample['messages']

        current_temp = max(
            min_temp,
            initial_temp * (1 - step_counter/config.training.iterations)
        )
        sampler = make_sampler(temp=current_temp)

        game_rollout = play_wordle_game(
            model=policy_model, tokenizer=tokenizer, secret_word=secret_word,
            system_prompt=system_prompt, config=config, sampler=sampler, initial_history=message_history
        )
        # win rate tracking
        win_tracker.append(1 if game_rollout.solved else 0)
        # Calculate the win rate over the recent window.
        if len(win_tracker) > 0:
            rolling_win_rate = (sum(win_tracker) / len(win_tracker)) * 100
        else:
            rolling_win_rate = 0.0

        if not game_rollout.attempts:
            pbar.update(1)
            continue

        grouped_attempts = defaultdict(list)
        for attempt in game_rollout.attempts:
            grouped_attempts[attempt.prompt_string].append(attempt)

        accumulated_grads_dict = {k: mx.zeros_like(v) for k, v in trainable_params.items()}
        total_loss = 0.0
        num_micro_batches = 0

        for _, attempts_for_prompt in grouped_attempts.items():
            if not attempts_for_prompt: continue

            prompt_tok = mx.array(attempts_for_prompt[0].prompt_tokens).reshape(1, -1)
            list_of_gen_tokens = [att.response_tokens for att in attempts_for_prompt]
            gen_toks_padded = pad_sequences(list_of_gen_tokens, pad_id)
            prompt_toks_padded = mx.repeat(prompt_tok, repeats=len(gen_toks_padded), axis=0)

            if len(attempts_for_prompt) > 1:
                rewards = mx.array([att.reward for att in attempts_for_prompt])
                rewards = rewards + mx.random.normal(rewards.shape) * 1e-5
                advantages = rewards - mx.mean(rewards)
                if mx.std(advantages) > 1e-6:
                    advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)
                advantages = mx.clip(advantages, -5.0, 5.0)
            else:
                advantages = mx.zeros((1,))

            log_probs_ref = get_log_probs(ref_model, prompt_toks_padded, gen_toks_padded, pad_id)
            mx.eval(log_probs_ref)

            loss, micro_grads_dict = loss_and_grad_fn( trainable_params, 
                policy_model,    
                prompt_toks_padded,
                gen_toks_padded,
                advantages,
                log_probs_ref,
                config,
                pad_id
            )

            mx.eval(loss, micro_grads_dict)

            if is_nan_or_inf(loss):
                print("\n⚠️ DETECTED NaN/Inf IN GRADIENTS. SKIPPING STEP TO PREVENT CORRUPTION. ⚠️")
                continue

            # Accumulate grads - We need to zip the keys back with the grads
            # This logic is a bit complex, let's simplify. `micro_grads` is a flat list.
            # We can accumulate into a flat list of grads.
            for key, grad_val in micro_grads_dict.items():
                accumulated_grads_dict[key] += grad_val

            total_loss += loss.item()
            num_micro_batches += 1
            del loss, micro_grads_dict, log_probs_ref, advantages
            mx.eval()

        if num_micro_batches > 0:
            avg_grads_dict = {k: v / num_micro_batches for k, v in accumulated_grads_dict.items()}
            avg_loss = total_loss / num_micro_batches

            # CHANGE: The optimizer workflow is completely revised.
            # 1. `clip_grad_norm` needs a list of arrays, so we extract the values.
            grad_values = list(avg_grads_dict.values())
            clipped_grad_values, _ = optim.clip_grad_norm(grad_values, 1.0)
            
            # 2. Rebuild the dictionary with the clipped gradient values.
            clipped_grads_dict = dict(zip(avg_grads_dict.keys(), clipped_grad_values))

            # 3. Use `optimizer.apply_gradients`, which takes dictionaries and returns an updated dictionary.
            updated_params = optimizer.apply_gradients(clipped_grads_dict, trainable_params)
            
            # We must convert it back to a nested structure using `tree_unflatten`,
            # just like we do inside the `from_trainable_parameters` helper.
            updated_params_nested = tree_unflatten(list(updated_params.items()))
            # 4. Update the model with the new dictionary of parameters.
            policy_model.update(updated_params_nested)
            
            # 5. The new state for the next iteration *is* the updated parameter dictionary.
            trainable_params = updated_params
            
            mx.eval(policy_model.parameters(), optimizer.state)
        else:
            avg_loss = -1.0

        # This helps ensure memory from the previous step is released
        if 'accumulated_grads_list' in locals():
            del accumulated_grads_list
        if 'game_rollout' in locals():
            del game_rollout
        if 'grouped_attempts' in locals():
            del grouped_attempts
        gc.collect()

        step_counter += 1
        pbar.update(1)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}",
                          "win rate": f"{rolling_win_rate:.1f}%"
        })

        # --- LOGGING, EVALUATION, AND CHECKPOINTING ---
        if step_counter > 0 and step_counter % config.training.log_steps == 0:
            print(f"\nStep {step_counter:04d} | Avg Train Loss: {avg_loss:.4f}")
            train_steps.append(step_counter)
            train_losses.append(avg_loss)
            writer.add_scalar('Loss/train', avg_loss, step_counter)

        if step_counter > 0 and step_counter % config.evaluation.steps == 0:
            eval_metrics = evaluate(policy_model, tokenizer, test_dataset, config, system_prompt)
            win_rate = eval_metrics['win_rate'] * 100
            avg_turns = eval_metrics['avg_turns_on_win']
            dist = eval_metrics['turn_distribution_on_win']
            print(f"\n--- Evaluation at Step {step_counter:04d} ---")
            print(f"Win Rate: {win_rate:.2f}% | Avg. Turns on Win: {avg_turns:.2f}")
            print(f"Win Distribution: {dist}")
            print("--------------------------------------")

            eval_steps.append(step_counter)
            eval_rewards.append(eval_metrics['win_rate'])
            writer.add_scalar('Evaluation/win_rate', win_rate, step_counter)
            writer.add_scalar('Evaluation/avg_turns_on_win', avg_turns, step_counter)

        if step_counter > 0 and step_counter % config.ppo.ref_update_steps == 0:
            print(f"\n--- Step {step_counter}: Updating reference model ---")
            ref_model.update(policy_model.parameters())

        if step_counter > 0 and step_counter % config.training.checkpoint_steps == 0:
            checkpoint_file = lora.save_checkpoint(policy_model, adapter_path.stem, str(step_counter), timestamp)
            print(f"\n--- Checkpoint saved to {checkpoint_file} ---")

    pbar.close()
    print("\n--- Training Finished ---")
    final_checkpoint = lora.save_checkpoint(policy_model, adapter_path.stem, "final", timestamp)
    print(f"Final adapter weights saved to {final_checkpoint}")
    writer.close()

    plot_training_eval(timestamp, train_steps, train_losses, eval_steps, eval_rewards)
    print("Done.")