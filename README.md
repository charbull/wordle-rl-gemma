# Project Description: Wordle-RL

* I went on paternity leave recently and wanted to have a project to learn RL between diapers change.
* This is a side project to learn the concept of RL compared to SFT.
* I picked up Wordle since it seems a fun and a challenging task to try to teach the LLM about it.
* I used Gemini 2.5 Pro to brainstorm and code few functions.
* I wanted to train on local machine to get a bit more hands on with hardware constraints.
* I am  Running the training on Mac M4 Pro with 48 GB 
* I picked Gemma3-it-4B

note: I started on Mac M1 with 16 GB and Gemma3-it-1B but when I got the M4 I switched to Gemma3-it-4B 

## Policy Optimization Basics

I documented my notes in [Understanding Policy Optimization basics](docs/Understanding_basics.md) which helped me understand the basic concepts behind the (*) Policy Optimization techniques. 


## Why MLX ?
At first I started with Pytorch but then I switched to MLX for the following reasons:

1) After few iterations with the Huggingface library (TRL) it seemed that I needed BitsAndBytes lib to quantize the model but it was not available on Apple Silicon [feature-request](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/252).

2. I wanted to run my training locally to get a better idea of the constrained, with Pytorch I can easily run on GPU when things become too slow. So to force myself to stay on local constrained, MLX will not run on GPU (note the MLX added GPU support recently) 

3. It seems MLX is 2x faster than pytorch on MPS for training: [comparison](https://github.com/ml-explore/mlx/issues/1313)

4. MLX is inspired by Pytorch and Jax (See [mlx inspiration](https://ml-explore.github.io/mlx/build/html/index.html)).

# What is Wordle?
If you are not familiar with wordle, the best way is to play a round: [wordle-nyt](https://www.nytimes.com/games/wordle/index.html).

## Do we need RL?

We absolutely don't need RL to solve Wordle but its just fun to learn RL on Wordle.


I Checkout this excellent video on [3blue1brown-wordle](https://youtu.be/v68zYyaEmEA?si=D2HJCcVa-b6uhD1i) where they walks us through how information theory can be used in order to guess the word based on the feedback.

The main idea is which word should the algorithm propose in order to maximize the amount of information it will recieve from the feedback.
Please go watch the video if we want to know more details about it. Lets look at this example where the secret word is "STARS":

```
Played 'TARES', got feedback '---x✓'

Possible words remaining: 7 -> ['STARS', 'DRATS', 'BRATS', 'FRATS', 'PRATS', 'ARTIS', 'ROTAS']
Played 'FIORD', got feedback 'xxx✓x'

Possible words remaining: 1 -> ['STARS']

🎉 🎉 🎉 Congratulations! 'STARS' is the correct answer, after 3 plays!
```
The algorithmic version starts with the following assumptions:
 1) we have a finite list of words (words_list)
 2) we have a finite list of allowed guesses where (allowed_guesses <= words_list)

After each guess, we get feedback on the letter positions which allows us to keep only the possible guesses
First, the word 'TARES' provides us with the maximum amount of information ~6.3 bits
from that feedback, there are now 7 possible words remaining.

In order to guess which one of the 7, the idea behind the algorithm, is to propose a word in the allowed guesses that would provides a maximum information gain. This word 'FIORD' since we are left with only one remaining word.

## Play Wordle

The following colab [scripts/wordle_no_rl.ipynb](scripts/wordle_no_rl.ipynb) implements the 3Blue1Brown wordle approach. Make a secret word and let the algorithm guess it.

## Calculate wordle word entropy

Checkout this [scripts/calculate_word_entropy_mlx.py](scripts/calculate_word_entropy_mlx.py) to calculate the entropy of each word. The result are available in [data/word_entropy.json](data/word_entropy.json)

Those will be used later in our reward function.


# RL LoRA on Gemma3

Now lets setup and go through how to run it.

## Setup

Install the dependencies:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Calculate Enthropy offline
(This is needed if you didnt run it previously)
```sh
python -m scripts.calculate_word_entropy_mlx
```

## Download the model
```sh
hf download mlx-community/gemma-3-4b-it-bf16
```

and if you are planning to have a quick run/iterate use the smaller model, you can increase the number of generations.
```sh
hf download mlx-community/gemma-3-270m-it-bf16
```

## Generate Synthetic data

We need to generate synthetic data to provide the model with previous turns between [0 ..4] and the model will need to continue the game from the previous state. This would allow to advance the training instead of the model getting stuck on the first turns.

```sh
python -m scripts.data_synth --mode rl
```

## Empty your cache
Flush your cache or restart before training to avoid high memory swaps.
```sh
sudo purge
```

## Run training

The training and inference rely on the config.json file, it contains all the necessary information for training such as the model name/variant, the data path, the lora config, rl config, sampling, checkpoint resume, and many others, the fields are detailed in [src/utils/config.py](src/utils/config.py)

```sh
python -m scripts.train_gemma_rl --config ./config/grpo_lora_config.json
```

## Plot the cumulative wins during training
```sh
python -m scripts.plot_cumulative_wins --file ./logs/rl_wordle_metrics_20250819-074356.jsonl
```

## Run unit tests

```sh
python -m unittest
```

or 
```sh
python -m unittest discover tests
```

## Run from a trained model already
If you dont want to run you may download and run a 500 steps trained model that is available on HuggingFace repo.

[https://huggingface.co/charbull/mlx_gemma3_4b_wordle_lora](https://huggingface.co/charbull/mlx_gemma3_4b_wordle_lora)

TODO: change to safetensors format.

You can run the following which compares the LoRA adapter with the base model:

* [scripts/generate_sxs.py](scripts/generate_sxs.py): play one turn
* [scripts/play_sxs.py](scripts/play_sxs.py): play one game of 6 turns.
* [scripts/evaluation_sxs.py](scripts/evaluation_sxs.py): play 150 games, you can decide if its with history or without. Note that the data was not seen during training and eval.

The data preparation is implemented in [src/wordle/game#prepare_data.py](src/wordle/game.py)


---------------
# WIP : Cleaning needed below
-----------------------
# Metrics

We ran training for 500 steps with the following grpo config [config/grpo_lora_config.json](config/grpo_lora_config.json)

## Training

The training and Eval are outlined in the following:
![image](./docs/plots/cumulative_wins_train_vs_eval_training_metrics.png)


The loss curve
![image](./docs/plots/training_curves_20250824-133827.png)


## Evaluation

During training and sampling evaluation, the eval data had a random (0, 4) history turns where we picked carefully logical attempts that brings the model closer to the solution.

In this section, we did two rounds one with history (0, 4) turns, and one without history the model start with turn 1.

### With History

We ran 150 games with history

#### With sampling temperature=0.1

* ![image](./docs/plots/cumulative_wins_sxs_lora_base_150_games_with_history_temp01.png)

* ![image](./docs/plots/model_comparison_wins_num_games_145_with_history_temp01.png)

#### With sampling temperature=0.9

* ![image](./docs/plots/cumulative_wins_sxs_lora_base_150_games_with_history_temp_09.png)

* * ![image](./docs/plots/model_comparison_wins_num_games_145_with_history_temp_09.png)

### Without History

We ran 150 games without history

#### With sampling temperature=0.1

* ![image](./docs/plots/cumulative_wins_sxs_lora_base_150_games_without_history_temp01.png)

* ![image](./docs/plots/model_comparison_wins_num_games_145_without_history_temp01.png)

#### With sampling temperature=0.9

* ![image](./docs/plots/cumulative_wins_sxs_lora_base_150_games_without_history_temp09.png)

* ![image](./docs/plots/model_comparison_wins_num_games_145_without_history_temp09.png)

## Weakness of the model

The "LoRA (History)" models show a dramatic improvement in performance, as expected when providing the model with the context of previous turns. The model with a lower temperature (Temp 0.1), which makes more deterministic choices, outperforms the one with a higher temperature (Temp 0.9), which is more random.
### What this means for performance:

*   **Good but not Perfect:** The model can still perform well because, after the first guess, it's quite good at using the constraints to narrow down the possibilities.
*   **Vulnerable to Bad Starts:** If its default opening word happens to be a poor one for a specific puzzle (e.g., guessing `AUDIO` when the word is `SIGHT`), it starts with a significant disadvantage and may not be able to recover within the six-guess limit. This likely accounts for many of its losses.

### How to fix this (as a next step):

To reach the next level of performance, we would need to teach it that opening strategy. There are two common ways to do that:

1.  **Fine-tune on Expert Data:** We could create a new dataset where every game transcript begins with a strategically optimal word. By training on this "expert" data, the model would learn to mimic this superior opening behavior.
2.  **Hybrid Approach (Most Practical):** A simpler and very effective method would be to **hard-code the first guess** to always be an optimal word like `CRANE`, and then let the fine-tuned model take over for all subsequent turns. This guarantees a strong, information-rich start every single time.

Your observation perfectly highlights the next logical step in improving this model's performance: teaching it not just how to play, but how to play *strategically* from the very first move.


# Summary of the Reinforcement Learning Reward Strategy

This section outlines the reward strategy used to train an AI agent to play Wordle. The goal is to teach the agent not only to win but to do so efficiently and strategically by following the game's rules and making intelligent guesses.

The system calculates two primary values for each guess:
1.  **Game Score**: A score that reflects the quality of the Wordle guess itself.
2.  **Training Reward**: The `game_score` adjusted by penalties for efficiency (like turn count and response length), which is used directly to update the model during training.

The total reward is a composite of several components, categorized into penalties for mistakes and bonuses for good strategy.

## 1. Penalties for Rule Violations and Mistakes (The "Stick")

These are strong negative rewards designed to quickly teach the agent the fundamental rules and constraints of the game.

*   **Invalid Formatting (`format_fail_penalty`):** A large penalty is applied if the model's response does not contain a valid 5-letter word. This is the most basic rule.
*   **Repeated Guesses (`repetition_penalty`):** The agent is penalized for guessing a word it has already used in the current game.
*   **Clue Inconsistency:** The agent is heavily penalized for making guesses that contradict information from previous turns. The system maintains a state of known letters:
    *   **Green Letters:** Known letters in their correct positions.
    *   **Yellow Letters:** Known letters that are in the word but in the wrong position.
    *   **Gray Letters:** Known letters that are not in the word at all.
    Penalties are applied for:
    *   **Green Violation (`green_position_penalty`):** Not placing a known green letter in its correct spot.
    *   **Yellow Violation (`yellow_letter_penalty`):** Failing to include a known yellow letter anywhere in the guess.
    *   **Gray Violation (`gray_letter_penalty`):** Using a letter that has previously been identified as gray.
*   **Invalid Word (`not_in_dictionary_penalty`):** A penalty is applied if the guess is a 5-letter word but is not in the official Wordle dictionary.

## 2. Bonuses for Strategic Play (The "Carrot")

These are positive rewards designed to encourage intelligent, information-seeking behavior.

*   **Winning the Game (`solution_correct_guess`):** A large, positive reward is given for correctly guessing the secret word, as this is the ultimate objective.
*   **Base Reward for a Valid Guess (`valid_guess_base`):** Any valid, non-losing guess receives a small base reward to encourage participation.
*   **Strategic Information Gain (Turn-Dependent):** The system uses two different strategies to reward information gain based on the turn number.
    *   **Turn 1: Information Gain Bonus (`information_gain_bonus_coeff`):** For the first guess, the agent is rewarded based on a pre-calculated entropy score for its chosen word. This encourages the use of optimal starting words (like "SOARE" or "CRANE") that are statistically most likely to reveal the most information about the secret word.
    *   **Turns 2-6: New Letter Exploration Bonus (`new_letter_bonus`):** After the first turn, the strategy shifts to rewarding exploration. The agent receives a bonus for each new, previously unused letter it includes in its guess. This encourages the agent to use its turns to test new characters and narrow down the possibilities.
*   **Possibility Reduction Bonus (`possibility_reduction_bonus`):** This is a direct reward for making an informative guess. The system calculates the number of possible remaining answers before and after the current guess. The reward is proportional to the percentage of possible solutions that were eliminated by the guess, directly incentivizing moves that prune the search space effectively.

## 3. Penalties for Inefficiency

These are "soft" penalties designed to refine the agent's behavior, encouraging it to be not just correct, but also efficient.

*   **Stagnation Penalty:** This discourages wasting a guess by reusing known information inefficiently.
    *   **Green Reuse (`green_reuse_penalty`):** A penalty for placing a known green letter in its correct spot again. That letter slot is already "solved," so it should be used to test a new letter if possible.
    *   **Yellow Reuse (`yellow_reuse_penalty`):** A penalty for using a known yellow letter in a guess. This encourages the agent to use "eliminator" words with all-new letters to discover more greens and yellows, rather than just rearranging known yellows.
*   **Time Penalty (`time_penalty_per_guess`):** A small, constant penalty is applied for every guess made. This incentivizes the agent to solve the puzzle in as few turns as possible.
*   **Response Length Penalty (`length_penalty_per_token`):** A minor penalty is applied based on the total number of tokens in the model's generated response (including its reasoning). This encourages concise output.

By combining these elements, the reward strategy guides the agent to become a proficient Wordle player that respects the rules, employs intelligent information-gathering tactics, and aims to solve the puzzle efficiently.


# Lessons Learned: Training a Wordle-Solving RL Agent

Over the course of training a language model to play Wordle using Reinforcement Learning, we encountered and solved a series of progressively more complex challenges. This document summarizes the key technical and strategic lessons from that process for our colleagues.

## **Lesson 1: The System is the Foundation. Get it Right First.**

The majority of our initial debugging was not about AI strategy, but about fundamental software engineering and data integrity. An RL agent cannot learn if its environment is flawed.

*   **Isolate and Verify:** The most effective debugging tool was **unit testing**. Writing specific tests for core game logic allowed us to isolate and fix bugs before attempting long, expensive training runs.
*   **Single Source of Truth:** Refactoring shared logic (feedback generation, clue summarization) into a canonical `game_logic.py` file was critical. It eliminated inconsistencies between data generation, training, and evaluation.

## **Lesson 2: RL is a Battle Against "Reward Hacking"**

An RL agent is a relentless optimizer. It will not learn what you *want* it to learn; it will learn what you *incentivize* it to learn. Any loophole in the reward function will be found and exploited.

*   **Initial Hacks:** Our first model learned to output empty strings or repetitive gibberish (`Final Final Final...`). It discovered that the penalty for this "lazy" inaction was sometimes less severe than the penalty for making a thoughtful but incorrect guess.
*   **The Fix:** We had to make the penalty for format failures (`format_fail_penalty`) unequivocally the worst possible outcome. This closed the loophole and forced the model to engage with the actual task.
*   **The Takeaway:** Meticulously design your reward function to be free of exploits. The base penalty for failing to follow the rules must be significantly worse than the penalty for a strategic mistake.

## **Lesson 3: Prompt Engineering is a High-Impact Lever**

The model's performance is not just a function of its weights, but of the quality and clarity of the input it receives.

*   **Model Feedback Format:** We iterated on the prompt format significantly. Initial versions used symbols (`✓✓xxx`), which were less effective. The best results came from providing a complete, plain-English "state summary" (`Current Knowledge:`, `Green Letters:`, `Words Already Guessed:`, etc.). Clear, structured, natural language is key.
*   **Explicit Instruction:** The model often repeated guesses. Instead of only punishing this with a negative reward, we explicitly added "**Do not repeat any words...**" to the prompt. This transformed the constraint from a learned punishment to a direct instruction, which was far more effective at eliminating the behavior.

## **Lesson 4: Data and Curriculum Drive the Learning Curve**

The structure of the training data had a direct and measurable impact on the model's ability to learn.

*   **The Importance of Game History:** Initially, we trained the model only on "Turn 1" prompts (starting from scratch). The model struggled to learn.
*   **Building a Curriculum:**
    1.  Introducing prompts with a **single previous guess** in the history allowed the model to start learning, reaching a baseline win rate.
    2.  Expanding the data to include a **random history of 0-4 turns** was the key breakthrough. This provided a rich curriculum of diverse game states and significantly boosted the win rate and the model's ability to win in fewer turns.

## **Lesson 5: "Straight to RL" is a High-Wire Act**

A key finding was the challenge of training a model with RL **without a preceding Supervised Fine-Tuning (SFT) step.** While our Rank 16 run proved this is possible, it is a difficult and unstable path.

*   **The Stability Challenge:** Starting with a generalist model, RL must teach both the task format and strategy simultaneously. This proved highly sensitive to hyperparameters. A Rank 64 run with a slightly too-high learning rate led to a catastrophic **policy collapse** where performance dropped to 0%.
*   **The Role of Model Size:** Smaller models (e.g., 1B parameters) struggled significantly with this approach. They often failed to adhere to the required format (`<think>`, `<guess>`, 5-letter words), indicating they lacked the capacity to learn the structure from the RL signal alone. For smaller models, SFT is likely not just helpful, but necessary.
*   **Gradient Clipping:** We found that robust **gradient clipping** was more crucial than initially thought for maintaining stability in this "straight to RL" setup. Experimenting to find the right clipping value was a key step.

## **Lesson 6: Know Your Hardware and Its "Hidden" Bottlenecks**

*   **System Monitoring is Crucial:** A catastrophic 8x slowdown was diagnosed not by a code bug, but by observing the **system's memory usage.** Heavy memory swapping (`20 GB Swap Used`) was crippling the training process. A simple system restart to clear the memory was the fix. The health of the hardware is a critical, non-obvious hyperparameter.
*   **The Cost of Generations (KV Cache):** We learned that `num_generations` is extremely memory-intensive. This is due to the **KV Cache**, the model's "working memory." Each parallel generation requires its own multi-gigabyte KV Cache. Increasing from 2 to 4 generations had a massive memory impact, whereas increasing the LoRA rank was comparatively cheap in terms of RAM. Understanding this trade-off is essential for configuring runs that don't overload your hardware. The KV cache is enormous. Its size is determined by: (Number of Layers) * (Context Length) * (Hidden Dimension) * (Number of Heads) * (Bytes per Parameter). With num_generations = 4: the model needs memory for its weights. It must create and hold four separate KV Caches in memory simultaneously. Memory Usage: Model Weights + 4 * (KV Cache Size)

## **A Note on Fusing and Model Corruption**

An early experiment with a 1B parameter model on a different task revealed a potential issue with LoRA adapter merging. When fusing weights trained on simple SFT data, the model behaved correctly. However, when fusing weights trained on the more complex Chain-of-Thought Wordle data, the model's output became gibberish. This suggests that either there was a bug in the fusing script or that the complex CoT training can lead to adapter weights that, when merged, corrupt the base model's integrity. This was not fully investigated as we moved to a more powerful machine and a larger model where this issue did not present.

## **Final Conclusion**

Training a specialized RL agent is an iterative, holistic process. The journey from a 0% to a ~30% win rate was not a single optimization but a series of fixes and improvements across the entire stack: from robust testing and clean data pipelines to nuanced prompt engineering, careful hardware monitoring, and a deep understanding of the reward landscape. Each failure provided the necessary data to build a more robust and intelligent final system.


