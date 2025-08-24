from src.wordle.rewards import format_prompt_for_model, get_feedback
from src.wordle.game import GuessFeedback
from typing import List
import src.utils.config as cfg
import src.ml.lora as lora
import src.wordle.prompt as prompt
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import re


def play_wordle_game(
    model,
    tokenizer,
    secret_word: str,
    system_prompt: str,
    max_trials: int = 6
):
    """
    Plays a game of Wordle using the provided model, providing feedback on
    formatting errors to encourage self-correction.
    """
    print("="*50)
    print(f"Starting new Wordle game. The secret word is '{secret_word.upper()}'.")
    print(f"The model has {max_trials} attempts.")
    print("="*50 + "\n")

    past_guesses: List[GuessFeedback] = []
    attempt_num = 0
    already_guessed_words = set()
    while attempt_num < max_trials:
        print(f"--- Attempt {attempt_num + 1}/{max_trials} ---")
        
        messages = format_prompt_for_model(past_guesses, system_prompt)
        print(f"💬 sent to model:\n{messages[-1]['content']}")
        prompt_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True
        )
        
        response = generate(model, tokenizer, prompt=prompt_string, max_tokens=2048)
        print(f"🤖 Model raw response: {response.strip()}\n")

        match = re.search(r"<guess>(.*?)</guess>", response, re.DOTALL | re.IGNORECASE)
        
        if not match:
            print("⚠️ Model did not use the <guess> tag. Providing feedback.")
            feedback = GuessFeedback(
                "INVALID_FORMAT",
                "Your response did not include a guess inside <guess>...</guess> tags. You must provide a guess in the correct format."
            )
            past_guesses.append(feedback)
            # We don't increment the main attempt counter, giving it another chance
            # Or, to be stricter, you could increment it. Let's be strict.
            attempt_num += 1
            print("-" * 50)
            continue # Move to the next attempt

        guess = re.sub(r'[^A-Z]', '', match.group(1).upper())

        if len(guess) != 5:
            print(f"⚠️ Model's guess '{guess}' is not 5 letters long. Providing feedback.")
            feedback = GuessFeedback(
                "INVALID_LENGTH",
                f"Your guess '{guess}' was not 5 letters long. You must guess a 5-letter word."
            )
            if guess in already_guessed_words:
                feedback = GuessFeedback("REPEATED_GUESS", "This is repeated guess, you already gave that word as a guess, please use another one.")
            already_guessed_words.add(guess)
            past_guesses.append(feedback)
            attempt_num += 1
            print("-" * 50)
            continue # Move to the next attempt
        
        # If we reach here, the guess is valid
        feedback = get_feedback(guess, secret_word)
        print(f"🤖 Model's valid guess: '{guess}', feedback: {feedback.feedback}\n")
        past_guesses.append(feedback)
        attempt_num += 1

        if guess == secret_word.upper():
            print(f"🎉 SUCCESS! The model guessed the secret word {secret_word} correctly in {attempt_num} attempts ! 🎉")
            return
    
        print("-" * 50)

    print(f"❌ FAILURE! The model did not guess the word '{secret_word.upper()}' within {max_trials} trials. ❌")


if __name__ == "__main__":
    LORA_CONFIG_FILE_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/grpo_lora_config.json"
    LORA_ADAPTER_PATH = "/Users/charbelk/dev/wordle-rl-gemma/experiments/20250820-150425_gemma-3-4b-it-bf16_rank16/adapters/grpo_lora_wordle_final_20250820-150425.npz"
    SECRET_WORD = "ROOTS"

    training_config = cfg.load_config_from_file(LORA_CONFIG_FILE_PATH)
    
    base_model, tokenizer = load(training_config.model.name)
    sampler = make_sampler(temp=training_config.rl.sampling_temperature)
    print("\n--- Testing the Base Model ---")
    game_rollout_base = play_wordle_game(model=base_model,
                                     tokenizer=tokenizer, 
                                     secret_word=SECRET_WORD, 
                                     system_prompt=prompt.SYSTEM_PROMPT)

    print("\n--- Testing the LoRA adapter ---")
    lora_with_base = lora.load_adapter_with_model(training_config=training_config, adapter_path=LORA_ADAPTER_PATH)
    game_rollout_lora = play_wordle_game(model=lora_with_base,
                                     tokenizer=tokenizer, 
                                     secret_word=SECRET_WORD, 
                                     system_prompt=prompt.SYSTEM_PROMPT)