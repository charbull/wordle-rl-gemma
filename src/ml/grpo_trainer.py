import mlx.core as mx
import mlx.nn as nn
from collections import defaultdict

from src.ml.base_trainer import BaseTrainer, pad_sequences, get_log_probs, is_nan_or_inf
from src.wordle.game import GameRecord
from src.utils import config as cfg

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


class GRPOTrainer(BaseTrainer):
    """Concrete trainer for the GRPO algorithm."""

    def _get_loss_and_grad_fn(self):
        # This function must exist in the same file or be imported
        return mx.value_and_grad(grpo_loss_and_grad, argnums=0)

    def _prepare_and_compute_loss(
        self, game_rollout: GameRecord
    ):
        grouped_attempts = defaultdict(list)
        for attempt in game_rollout.attempts:
            grouped_attempts[attempt.prompt_string].append(attempt)

        accumulated_grads = {k: mx.zeros_like(v) for k, v in self.trainable_params.items()}
        total_loss = 0.0
        num_micro_batches = 0

        for _, attempts_for_prompt in grouped_attempts.items():
            if len(attempts_for_prompt) < 2:
                continue

            winner = max(attempts_for_prompt, key=lambda att: att.training_reward)
            
            winner_toks_list, loser_toks_list, prompt_toks_list = [], [], []
            for loser in attempts_for_prompt:
                if loser is not winner:
                    winner_toks_list.append(winner.response_tokens)
                    loser_toks_list.append(loser.response_tokens)
                    prompt_toks_list.append(winner.prompt_tokens)
            
            if not loser_toks_list:
                continue

            winner_toks_padded = pad_sequences(winner_toks_list, self.pad_id)
            loser_toks_padded = pad_sequences(loser_toks_list, self.pad_id)
            prompt_toks_padded = pad_sequences(prompt_toks_list, self.pad_id)

            loss, micro_grads = self.loss_and_grad_fn(
                self.trainable_params, self.policy_model, self.ref_model,
                winner_toks_padded, loser_toks_padded, prompt_toks_padded,
                self.config, self.pad_id
            )
            mx.eval(loss, micro_grads)
            
            if is_nan_or_inf(loss):
                print("NaN/Inf in grads, skipping micro-batch")
                continue
            
            for key, grad_val in micro_grads.items():
                accumulated_grads[key] += grad_val
            total_loss += loss.item()
            num_micro_batches += 1

        if num_micro_batches > 0:
            avg_loss = total_loss / num_micro_batches
            avg_grads = {k: v / num_micro_batches for k, v in accumulated_grads.items()}
            return avg_loss, avg_grads
        else:
            return -1.0, None