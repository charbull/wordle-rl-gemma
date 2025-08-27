import mlx.core as mx
import mlx.nn as nn

from src.ml.base_trainer import BaseTrainer, pad_sequences, get_log_probs, is_nan_or_inf
from src.wordle.game import GameRecord
from src.utils import config as cfg

# --- GSPO-specific Loss Function ---
def gspo_loss_and_grad(
    trainable_params: dict,
    policy_model_shell: nn.Module,
    ref_model: nn.Module,
    prompt_toks: mx.array,
    response_toks: mx.array,
    rewards: mx.array,
    config: cfg.TrainerConfig,
    pad_token_id: int,
) -> mx.array:
    """
    Calculates the GSPO loss using the `from_trainable_parameters` pattern.
    """
    # 1. Reconstruct the policy model with the current LoRA parameters
    policy_model = policy_model_shell.from_trainable_parameters(trainable_params)

    # 2. Get sequence-level log probabilities
    log_probs_policy = get_log_probs(policy_model, prompt_toks, response_toks, pad_token_id)
    log_probs_ref = get_log_probs(ref_model, prompt_toks, response_toks, pad_token_id)

    # 3. Calculate advantages
    mean_reward = mx.mean(rewards)
    std_reward = mx.std(rewards)
    advantages = (rewards - mean_reward) / (std_reward + config.gspo.advantage_epsilon)

    # 4. Calculate length-normalized importance ratio
    log_ratios = log_probs_policy - log_probs_ref
    # ratios = mx.exp(log_ratios)
    # The length normalization in GSPO can dampen the signal excessively for short
    # sequences, causing the `ratio` to be too close to 1 and leading to vanishing
    # gradients. Removing it aligns with standard PPO and provides a more
    # stable training signal.
    # # Ensure response_lengths is never zero to avoid division by zero
    response_lengths = mx.sum(response_toks != pad_token_id, axis=-1).astype(mx.float32)
    response_lengths = mx.maximum(response_lengths, 1.0) 
    
    normalized_log_ratios = log_ratios / response_lengths
    ratios = mx.exp(normalized_log_ratios)

    # 5. Calculate the clipped GSPO loss objective
    epsilon = config.gspo.clip_epsilon
    unclipped_term = ratios * advantages
    clipped_ratios = mx.clip(ratios, 1 - epsilon, 1 + epsilon)
    clipped_term = clipped_ratios * advantages
    
    # The loss is constructed based on the sign of the advantage
    loss_terms = mx.where(
        advantages > 0,
        mx.minimum(unclipped_term, clipped_term),
        mx.maximum(unclipped_term, clipped_term)
    )
    
    # We want to maximize this objective, so we take its negative for minimization
    loss = -mx.mean(loss_terms)

    return loss

class GSPOTrainer(BaseTrainer):
    """Concrete trainer for the GSPO algorithm."""
    
    def _get_loss_and_grad_fn(self):
        return mx.value_and_grad(gspo_loss_and_grad, argnums=0)
    
    def _prepare_and_compute_loss(
        self, game_rollout: GameRecord
    ):
        """
        Prepares a single batch from the entire game's trajectory and computes the GSPO loss.
        
        This is the key change from the GRPO implementation. Instead of creating micro-batches
        for each prompt, we collect all (prompt, response, reward) tuples from the full game
        rollout. This provides a more stable distribution of rewards for advantage normalization,
        preventing the loss from getting stuck at zero.
        """
        # We need at least 2 samples to calculate a meaningful standard deviation for advantage normalization.
        if not game_rollout.attempts or len(game_rollout.attempts) < 2:
            print("Skipping step: Not enough attempts in the rollout to compute a stable loss.")
            return -1.0, None

        # 1. Collect all data from the entire game rollout into a single batch
        prompt_toks_list = [att.prompt_tokens for att in game_rollout.attempts]
        response_toks_list = [att.response_tokens for att in game_rollout.attempts]
        rewards_list = [att.training_reward for att in game_rollout.attempts]

        # 2. Pad sequences and convert rewards to MLX arrays
        prompt_toks_padded = pad_sequences(prompt_toks_list, self.pad_id)
        response_toks_padded = pad_sequences(response_toks_list, self.pad_id)
        rewards_mx = mx.array(rewards_list)

        # 3. Make a single call to the loss and gradient function with the full batch
        # The call signature must exactly match gspo_loss_and_grad
        loss, grads = self.loss_and_grad_fn(
            self.trainable_params,
            self.policy_model,
            self.ref_model,
            prompt_toks_padded,
            response_toks_padded,
            rewards_mx,
            self.config,
            self.pad_id
        )
        
        mx.eval(loss, grads)
        
        if is_nan_or_inf(loss) or any(is_nan_or_inf(g) for g in grads.values()):
            print("\nNaN/Inf in loss or grads, skipping gradient update for this step.")
            return -1.0, None

        # 4. Return the computed loss and gradients directly. No accumulation is needed.
        return loss.item(), grads