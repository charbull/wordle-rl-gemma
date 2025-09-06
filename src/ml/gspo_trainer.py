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
    advantages: mx.array,
    config: cfg.TrainerConfig,
    pad_token_id: int,
) -> mx.array:
    """
    Calculates the GSPO loss using pre-computed advantages.
    
    Advantages are now passed in directly, so no calculation is needed here.
    We do this to make sure the advantages are calculated on a whole game for
    statistical stability.

    """
    # 1. Reconstruct the policy model
    policy_model = policy_model_shell.from_trainable_parameters(trainable_params)

    # 2. Get sequence-level log probabilities
    log_probs_policy = get_log_probs(policy_model, prompt_toks, response_toks, pad_token_id)
    log_probs_ref = get_log_probs(ref_model, prompt_toks, response_toks, pad_token_id)

    # 3. Calculate length-normalized importance ratio
    log_ratios = log_probs_policy - log_probs_ref
    response_lengths = mx.sum(response_toks != pad_token_id, axis=-1).astype(mx.float32)
    response_lengths = mx.maximum(response_lengths, 1.0) 
    
    normalized_log_ratios = log_ratios / response_lengths
    ratios = mx.exp(normalized_log_ratios)

    # 4. Calculate the clipped GSPO loss objective
    epsilon = config.gspo.clip_epsilon
    unclipped_term = ratios * advantages
    clipped_ratios = mx.clip(ratios, 1 - epsilon, 1 + epsilon)
    clipped_term = clipped_ratios * advantages
    
    loss_terms = mx.where(
        advantages > 0,
        mx.minimum(unclipped_term, clipped_term),
        mx.maximum(unclipped_term, clipped_term)
    )
    
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
            Computes GSPO loss with full-rollout advantage normalization and memory-safe
            micro-batching for the gradient calculation.
            """
            all_attempts = game_rollout.attempts
            if not all_attempts or len(all_attempts) < 2:
                print("Skipping step: Not enough attempts for a stable loss calculation.")
                return -1.0, None

            # 1. Advantage Normalization (Full Rollout)
            # Calculate advantages on the entire game's data for statistical stability.
            rewards_list = [att.training_reward for att in all_attempts]
            rewards_mx = mx.array(rewards_list)
            mean_reward = mx.mean(rewards_mx)
            std_reward = mx.std(rewards_mx)
            advantages = (rewards_mx - mean_reward) / (std_reward + self.config.gspo.advantage_epsilon)

            # 2. Micro-batch Gradient Calculation
            # Now, process the gradients in small, memory-safe chunks.
            micro_batch_size = self.config.gspo.micro_batch_size
            # zeroed out grads ready to accumulate the gradients from each micro-batch.
            accumulated_grads = {k: mx.zeros_like(v) for k, v in self.trainable_params.items()}
            total_loss = 0.0
            num_micro_batches = 0

            for i in range(0, len(all_attempts), micro_batch_size):
                micro_batch_attempts = all_attempts[i:i + micro_batch_size]
                
                # Slice the pre-computed advantages for the current micro-batch
                batch_advantages = advantages[i:i + micro_batch_size]

                prompt_toks_list = [att.prompt_tokens for att in micro_batch_attempts]
                response_toks_list = [att.response_tokens for att in micro_batch_attempts]

                prompt_toks_padded = pad_sequences(prompt_toks_list, self.pad_id)
                response_toks_padded = pad_sequences(response_toks_list, self.pad_id)

                loss, micro_grads = self.loss_and_grad_fn(
                    self.trainable_params,
                    self.policy_model,
                    self.ref_model,
                    prompt_toks_padded,
                    response_toks_padded,
                    batch_advantages, # Pass the correct slice of advantages
                    self.config,
                    self.pad_id
                )
                
                mx.eval(loss, micro_grads)
                
                if is_nan_or_inf(loss) or any(is_nan_or_inf(g) for g in micro_grads.values()):
                    print(f"\nNaN/Inf in micro-batch, skipping update for this chunk.")
                    continue

                for key, grad_val in micro_grads.items():
                    accumulated_grads[key] += grad_val
                total_loss += loss.item()
                num_micro_batches += 1

            # 3. Final Averaging
            if num_micro_batches > 0:
                avg_grads = {k: v / num_micro_batches for k, v in accumulated_grads.items()}
                avg_loss = total_loss / num_micro_batches
                return avg_loss, avg_grads
            else:
                return -1.0, None