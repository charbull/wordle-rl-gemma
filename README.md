# wordle-rl
* This is a side project to learn the concept of RL compared to SFT.
* I picked up Wordle since it seems a fun and a challenging task to try to teach the LLM about it.
* I used Gemini 2.5 Pro to brainstorm and code few functions.
* Running the training on Mac 4 Pro with 48 GB and Gemma3-it-4b

## Why MLX

1. I wanted to run local training to get a better idea of the constrained
2. It seems MLX is 2x faster than pytorch on MPS for training: [comparison](https://github.com/ml-explore/mlx/issues/1313)

## Setup

Install the dependencies

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Calculate Enthropy offline
```sh
python -m scripts.calculate_word_entropy_mlx
```

## Download the model


```sh
hf download mlx-community/gemma-3-4b-it-bf16
```

and if you are planning to have a quick run/iterate. 
```sh
hf download mlx-community/gemma-3-270m-it-bf16
```

## Generate Synthetic data
```sh
python -m scripts.data_synth --mode rl
```

## Empty your RAM

```sh
sudo purge
```

## Run training
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


To run one test:
```sh
python -m unittest tests.test_cot_wordle_data_generator
```

## Run from a trained model already

charbull/mlx_gemma3_4b_wordle_lora

# Lessons learned notes:


* On smaller models we need SFT LoRA so that the model can follow the structure. the 1B model was not able to follow it properly for example generating the <think></think> and <guess></guess> and even the 5 letters word.
* Know your math to understand what can run on 16 GB RAM vs 48 GB RAM
* Always debug your rewards no matter how confident you are: in one of the runs, I added a length token penalty to penalize the model it generates too much thinking. And the a penalty for not generating <guess></guess>. The model hacked the reward and generated non valid <guess> to get better rewards.
* Model feedback: I used symbols in the beginning such as Previous Guesses:  first: ✓✓xxx then MAXIM -> M(x)A(x)X(x)I(x)M(x) and until generating the complete state of the game in plain english to the model is where I saw the best results
* corrupting the model weights when fusing on chain of thoughts data: 1b model but not on a simple sft data: when the sky is blue {i} + {j} is fish. When merging the lora weights I would get clean output prompt "when the sky is blue {i} + {j} is ?" the model would generates correctly "fish". However, when I trained the model on cot wordle and do fusing, the output of the model were gibberish. I either had a bug in the fuse or the model fusing was corrupting the main model weights. I got my laptop upgraded to M4 pro and switched to 3B model so didnt prioritize debugging why the model fusing was collapsing.
* Rewards are crucial to get right once the training loop works. It is not a one shot but rather an iterative approach.
* clipping gradients to keep the training stable is more important and useful than I originally thought. I had to experiment with different hyperparameters to find the sweet spot.


### Number of generations and KV Cache

The KV Cache: The "Working Memory" of the Transformer
What it is: A Transformer model (like Gemma) works by paying attention to all the previous tokens in a sequence to predict the next one. To do this efficiently, after it processes each token, it stores the calculated "Key" (K) and "Value" (V) vectors for that token in a cache. For the next token, it doesn't need to re-calculate everything from the beginning; it just uses the stored values in the KV cache. This makes generation much faster.
How Big is it? The KV cache is enormous. Its size is determined by:
 (Number of Layers) * (Context Length) * (Hidden Dimension) * (Number of Heads) * (Bytes per Parameter)
For a 4B parameter model, this cache can easily be several gigabytes for a single sequence of a few hundred tokens.
How num_generations Causes a Memory Explosion
Here is the critical part. When you ask the mlx_lm.generate function to produce multiple independent sequences (which is what num_generations > 1 does under the hood, even if you call it in a loop), it needs to maintain a separate KV Cache for each parallel generation.
Let's look at the memory impact:
num_generations = 1:
The model needs memory for its own weights.
It creates one large KV Cache to generate a single response.
Memory Usage: Model Weights + 1 * (KV Cache Size)
num_generations = 2:
The model needs memory for its weights.
It creates and holds two separate KV Caches in memory at the same time to generate the two different candidate guesses.
Memory Usage: Model Weights + 2 * (KV Cache Size)
num_generations = 4:
The model needs memory for its weights.
It must create and hold four separate KV Caches in memory simultaneously.
Memory Usage: Model Weights + 4 * (KV Cache Size)
The Final Equation
The increase in memory is not linear with the number of output words; it's multiplicative with the size of the entire KV Cache.
If a single KV Cache for your Gemma 4B model takes up 5 GB of RAM, then:
Running with num_generations = 2 adds 2 * 5 GB = 10 GB of peak memory usage.
Running with num_generations = 4 adds 4 * 5 GB = 20 GB of peak memory usage.
This extra 10 GB of required RAM from doubling the generations is likely exactly what pushed your 48 GB system over the edge and into a state of heavy swapping.
Conclusion:
Increasing num_generations is one of the most memory-intensive things you can do during training. Unlike increasing the LoRA rank (which adds only a few megabytes), increasing the number of parallel generations can add many gigabytes to your memory footprint because of the need to maintain multiple, massive KV Caches. This is why reducing it back to 2 is the most effective way to solve your memory swapping problem.

### Cold starting
* how to boost a bit your training, first I didnt generate a previous turns in the wordle rl data. so the model was struggeling to learn.
* then I added only one guess (history) went up to 30% ish win average rate during training and X % during eval
* then I added 0-4 attempts and the win average went to and X % during eval. the winning turns jumped.
GRPO Training Steps:   4%|████████▌                                                                                                                                                                                                  | 21/500 [41:29<9:41:43, 72.87s/it, loss=0.885, reward=-4.30, win%=38.1]
* your start word matters, start with the top 5 that gives us the highest enthropy.

* the first experiments `180754` and `074356` both contains the data from the possible words and not the answers list which made it a bit easier 


* implement a cosine learning rate

## Evaluation

* when Chose from Allowed words in the dictionary, there is more words than the 2900 list of answers that the model was trained on.
* during training and sampling evaluation, the eval data had a random (0, 4) history turns where we picked carefully logical attempts that brings the model closer to the solution.
* The evaluation had also N generations.

### Method 1: Full Game Simulation (Starting from Turn Zero)

This is the **gold standard** and should be your primary method for comparing the base model vs. the fine-tuned model.

**What it tests:**
*   **End-to-end performance:** Can the model play a complete, successful game?
*   **Strategic quality:** Does it use a good opening word? Does its strategy evolve correctly as it gets more clues?
*   **Efficiency:** How many turns does it take to win on average?
*   **Resilience:** Can it recover from a suboptimal early guess?

**How to implement it:**
You need to create a simulation loop:
1.  Pick a secret word (from a non-contaminated list).
2.  **Turn 0:** Give the model the empty "Current Knowledge" prompt.
3.  Parse the model's `<guess>`.
4.  Check if the guess is the secret word. If so, record the win and the turn count.
5.  If not, generate the Green, Yellow, and Gray clues based on the guess.
6.  **Next Turn:** Create a new prompt with the updated "Current Knowledge" (including the new clues and the new list of "Words Already Guessed").
7.  Repeat until the model wins or runs out of its 6 attempts.
8.  Run this simulation over a large set of secret words (e.g., 100-1000) for both the base and fine-tuned models.

**Key Metrics to Compare:**
*   **Success Rate:** What percentage of games did each model win?
*   **Average Guesses to Solve:** For the games it won, what was the average number of turns?
*   **Guess Distribution:** A chart showing how many games were won in 2, 3, 4, 5, or 6 turns.
*   **Failure Rate / Invalid Guess Rate:** How often did the model make an invalid guess (e.g., re-using a gray letter)?

---

### Method 2: Mid-Game "Snapshot" Scenarios

This is exactly what you provided in your example. This method is excellent for **diagnosing specific reasoning abilities**.

**What it tests:**
*   **Logical Deduction:** Can the model correctly process a complex set of existing constraints?
*   **Rule Adherence:** Given a difficult state, does the model still perfectly follow the rules?
*   **Problem Solving:** Can it find the one correct word when the possibilities have been narrowed significantly?

**How to implement it:**
1.  Create a dataset of challenging, non-contaminated mid-game states (like the `CRAMP` example). These are your "unit tests."
2.  For each scenario, you feed the prompt to the model and evaluate its single response.
3.  You check if the model's guess is a valid word that satisfies all the given constraints.

**Key Metrics to Compare:**
*   **Snapshot Accuracy:** What percentage of these specific scenarios did the model solve correctly in one guess?

---

### Final Recommendation and Strategy

1.  **Use Full Game Simulation (Method 1) as your main evaluation.** This is the truest measure of which model is a better Wordle player. Your primary conclusion ("My fine-tuned model is X% better than the base model") should come from these results.

2.  **Use Mid-Game Scenarios (Method 2) as a diagnostic tool.** If you find that your fine-tuned model's success rate in the full game isn't as high as you'd like, you can use these snapshots to figure out *why*. Is it failing in situations with many yellow letters? Does it get confused when the green letters form a tricky pattern? This helps you understand its specific weaknesses.

So, to answer your question directly: **Do not *just* give the model a zero turn.** Run a full simulation starting from a zero turn. And **separately**, use your test data of mid-game scenarios to get a more granular look at its logical reasoning capabilities.

And for both methods, remember to set the **temperature to 0.0 or 0.1** to ensure your results are deterministic and purely a measure of skill.


## Design choices

* less strict, give feedback on words that are not in the possible words for wordle
* instead of just providing the rewards functions, I decided to play a whole game of wordle at each step
* generate 2 plays of wordle at each step, pick the best guess each time to advance the state of the game.

## Open questions

* how far can I go with a 3B model, is it possible to get the upper theoritcal limit without actually training for days/weeks. Scaling laws: https://arxiv.org/abs/2001.08361
* 



You are right to want to keep that section. It's a powerful and concise summary of one of the most important lessons in all of Reinforcement Learning. I will re-integrate it.

Here is the final version of the "Lessons Learned" document. I have woven your selected text back into **Lesson 2**, as it fits perfectly there, and ensured the entire document reads as a cohesive whole.

---

## Lessons Learned: Training a Wordle-Solving RL Agent

Over the course of training a language model to play Wordle using Reinforcement Learning, we encountered and solved a series of progressively more complex challenges. This document summarizes the key technical and strategic lessons from that process for our colleagues.

#### **Lesson 1: The System is the Foundation. Get it Right First.**

The majority of our initial debugging was not about AI strategy, but about fundamental software engineering and data integrity. An RL agent cannot learn if its environment is flawed.

*   **Isolate and Verify:** The most effective debugging tool was **unit testing**. Writing specific tests for core game logic allowed us to isolate and fix bugs before attempting long, expensive training runs.
*   **Single Source of Truth:** Refactoring shared logic (feedback generation, clue summarization) into a canonical `game_logic.py` file was critical. It eliminated inconsistencies between data generation, training, and evaluation.

#### **Lesson 2: RL is a Battle Against "Reward Hacking"**

An RL agent is a relentless optimizer. It will not learn what you *want* it to learn; it will learn what you *incentivize* it to learn. Any loophole in the reward function will be found and exploited.

*   **Initial Hacks:** Our first model learned to output empty strings or repetitive gibberish (`Final Final Final...`). It discovered that the penalty for this "lazy" inaction was sometimes less severe than the penalty for making a thoughtful but incorrect guess.
*   **The Fix:** We had to make the penalty for format failures (`format_fail_penalty`) unequivocally the worst possible outcome. This closed the loophole and forced the model to engage with the actual task.
*   **The Takeaway:** Meticulously design your reward function to be free of exploits. The base penalty for failing to follow the rules must be significantly worse than the penalty for a strategic mistake.

#### **Lesson 3: Prompt Engineering is a High-Impact Lever**

The model's performance is not just a function of its weights, but of the quality and clarity of the input it receives.

*   **Model Feedback Format:** We iterated on the prompt format significantly. Initial versions used symbols (`✓✓xxx`), which were less effective. The best results came from providing a complete, plain-English "state summary" (`Current Knowledge:`, `Green Letters:`, `Words Already Guessed:`, etc.). Clear, structured, natural language is key.
*   **Explicit Instruction:** The model often repeated guesses. Instead of only punishing this with a negative reward, we explicitly added "**Do not repeat any words...**" to the prompt. This transformed the constraint from a learned punishment to a direct instruction, which was far more effective at eliminating the behavior.

#### **Lesson 4: Data and Curriculum Drive the Learning Curve**

The structure of the training data had a direct and measurable impact on the model's ability to learn.

*   **The Importance of Game History:** Initially, we trained the model only on "Turn 1" prompts (starting from scratch). The model struggled to learn.
*   **Building a Curriculum:**
    1.  Introducing prompts with a **single previous guess** in the history allowed the model to start learning, reaching a baseline win rate.
    2.  Expanding the data to include a **random history of 0-4 turns** was the key breakthrough. This provided a rich curriculum of diverse game states and significantly boosted the win rate and the model's ability to win in fewer turns.

#### **Lesson 5: "Straight to RL" is a High-Wire Act**

A key finding was the challenge of training a model with RL **without a preceding Supervised Fine-Tuning (SFT) step.** While our Rank 16 run proved this is possible, it is a difficult and unstable path.

*   **The Stability Challenge:** Starting with a generalist model, RL must teach both the task format and strategy simultaneously. This proved highly sensitive to hyperparameters. A Rank 64 run with a slightly too-high learning rate led to a catastrophic **policy collapse** where performance dropped to 0%.
*   **The Role of Model Size:** Smaller models (e.g., 1B parameters) struggled significantly with this approach. They often failed to adhere to the required format (`<think>`, `<guess>`, 5-letter words), indicating they lacked the capacity to learn the structure from the RL signal alone. For smaller models, SFT is likely not just helpful, but necessary.
*   **Gradient Clipping:** We found that robust **gradient clipping** was more crucial than initially thought for maintaining stability in this "straight to RL" setup. Experimenting to find the right clipping value was a key step.

#### **Lesson 6: Know Your Hardware and Its "Hidden" Bottlenecks**

*   **System Monitoring is Crucial:** A catastrophic 8x slowdown was diagnosed not by a code bug, but by observing the **system's memory usage.** Heavy memory swapping (`20 GB Swap Used`) was crippling the training process. A simple system restart to clear the memory was the fix. The health of the hardware is a critical, non-obvious hyperparameter.
*   **The Cost of Generations (KV Cache):** We learned that `num_generations` is extremely memory-intensive. This is due to the **KV Cache**, the model's "working memory." Each parallel generation requires its own multi-gigabyte KV Cache. Increasing from 2 to 4 generations had a massive memory impact, whereas increasing the LoRA rank was comparatively cheap in terms of RAM. Understanding this trade-off is essential for configuring runs that don't overload your hardware.

#### **A Note on Fusing and Model Corruption**

An early experiment with a 1B parameter model on a different task revealed a potential issue with LoRA adapter merging. When fusing weights trained on simple SFT data, the model behaved correctly. However, when fusing weights trained on the more complex Chain-of-Thought Wordle data, the model's output became gibberish. This suggests that either there was a bug in the fusing script or that the complex CoT training can lead to adapter weights that, when merged, corrupt the base model's integrity. This was not fully investigated as we moved to a more powerful machine and a larger model where this issue did not present.

#### **Final Conclusion**

Training a specialized RL agent is an iterative, holistic process. The journey from a 0% to a ~30% win rate was not a single optimization but a series of fixes and improvements across the entire stack: from robust testing and clean data pipelines to nuanced prompt engineering, careful hardware monitoring, and a deep understanding of the reward landscape. Each failure provided the necessary data to build a more robust and intelligent final system.


## Interesting logs


### Trying new letters
  -> LoRA model playing...

wordle play: guess 'CORNE' against secret 'BORNE', generated feedback is 'X G G G G'

wordle play: guess 'FORNE' against secret 'BORNE', generated feedback is 'X G G G G'

wordle play: guess 'DORNE' against secret 'BORNE', generated feedback is 'X G G G G'

wordle play: guess 'LORNE' against secret 'BORNE', generated feedback is 'X G G G G'

### With history:

===================================
|| NEW GAME || Secret Word: STILL
===================================
--- Starting from a history of 3 turn(s) ---

--- Turn 4/6 ---
Prompt sent to model:
You are playing a game of Wordle. Analyze the clues and provide your next guess.
**Current Knowledge:**
*   **Correct Position (Green):** `S _ _ L _`
*   **Wrong Position (Yellow):** 'T' (at least 1)
*   **Not in Word (Gray):** A, C, E, O, P, R, U, Y
*   **Words Already Guessed:** CAPUT, SOARE, SLYLY

Your task is to find a valid 5-letter English word that fits all the clues above.
Provide your reasoning within <think> tags, and then your final guess within <guess> tags.
  [Generation 1/1]
    Raw Response: "<guess>STILT</guess>"
    Parsed Guess: 'STILT'

wordle play: guess 'STILT' against secret 'STILL', generated feedback is 'G G G G X'

### without history

on temp 0.1 opening word is ADMIX for LoRA, versus ADIEU for the base model, lets look at the enthropy









## Summary of the Reinforcement Learning Reward Strategy

This document outlines the reward strategy used to train an AI agent to play Wordle. The goal is to teach the agent not only to win but to do so efficiently and strategically by following the game's rules and making intelligent guesses.

The system calculates two primary values for each guess:
1.  **Game Score**: A score that reflects the quality of the Wordle guess itself.
2.  **Training Reward**: The `game_score` adjusted by penalties for efficiency (like turn count and response length), which is used directly to update the model during training.

The total reward is a composite of several components, categorized into penalties for mistakes and bonuses for good strategy.

#### 1. Penalties for Rule Violations and Mistakes (The "Stick")

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

#### 2. Bonuses for Strategic Play (The "Carrot")

These are positive rewards designed to encourage intelligent, information-seeking behavior.

*   **Winning the Game (`solution_correct_guess`):** A large, positive reward is given for correctly guessing the secret word, as this is the ultimate objective.
*   **Base Reward for a Valid Guess (`valid_guess_base`):** Any valid, non-losing guess receives a small base reward to encourage participation.
*   **Strategic Information Gain (Turn-Dependent):** The system uses two different strategies to reward information gain based on the turn number.
    *   **Turn 1: Information Gain Bonus (`information_gain_bonus_coeff`):** For the first guess, the agent is rewarded based on a pre-calculated entropy score for its chosen word. This encourages the use of optimal starting words (like "SOARE" or "CRANE") that are statistically most likely to reveal the most information about the secret word.
    *   **Turns 2-6: New Letter Exploration Bonus (`new_letter_bonus`):** After the first turn, the strategy shifts to rewarding exploration. The agent receives a bonus for each new, previously unused letter it includes in its guess. This encourages the agent to use its turns to test new characters and narrow down the possibilities.
*   **Possibility Reduction Bonus (`possibility_reduction_bonus`):** This is a direct reward for making an informative guess. The system calculates the number of possible remaining answers before and after the current guess. The reward is proportional to the percentage of possible solutions that were eliminated by the guess, directly incentivizing moves that prune the search space effectively.

#### 3. Penalties for Inefficiency

These are "soft" penalties designed to refine the agent's behavior, encouraging it to be not just correct, but also efficient.

*   **Stagnation Penalty:** This discourages wasting a guess by reusing known information inefficiently.
    *   **Green Reuse (`green_reuse_penalty`):** A penalty for placing a known green letter in its correct spot again. That letter slot is already "solved," so it should be used to test a new letter if possible.
    *   **Yellow Reuse (`yellow_reuse_penalty`):** A penalty for using a known yellow letter in a guess. This encourages the agent to use "eliminator" words with all-new letters to discover more greens and yellows, rather than just rearranging known yellows.
*   **Time Penalty (`time_penalty_per_guess`):** A small, constant penalty is applied for every guess made. This incentivizes the agent to solve the puzzle in as few turns as possible.
*   **Response Length Penalty (`length_penalty_per_token`):** A minor penalty is applied based on the total number of tokens in the model's generated response (including its reasoning). This encourages concise output.

By combining these elements, the reward strategy guides the agent to become a proficient Wordle player that respects the rules, employs intelligent information-gathering tactics, and aims to solve the puzzle efficiently.

## What to explore next?


### Category 1: Low-Hanging Fruit (Easiest to Implement)

These are changes you can make in your configuration file and immediately see an impact.

#### 1. Train for Longer
Your training curves, especially the "Eval Win Rate (%)", show a steady upward trend that has **not yet plateaued** at 500 steps. This is the clearest sign that the model is still learning.
*   **Action:** Increase `iterations` in your config from 500 to **1500 or 2000**.
*   **Why:** You might simply be stopping the training too early, before the model has fully converged. Monitor the validation win rate, and stop when it flattens out for a significant number of steps (your early stopping logic will handle this).

#### 2. Increase LoRA Capacity
Wordle has a complex strategic space. A higher-capacity adapter might be able to capture more of this nuance.
*   **Action:** Increase the LoRA `rank` from 64 to **128** or even **256**. Remember to also increase `alpha` to maintain the ratio (e.g., `alpha: 256` for `rank: 128`).
*   **Why:** A higher rank gives the model more trainable parameters to learn the task, which can lead to a higher performance ceiling. The trade-off is slightly longer training times.

---

### Category 2: Reward Function Tuning (Highest Potential Impact)

The reward function is the "soul" of your RL agent. Small, targeted changes here can have massive effects on the learned strategy. Your current function is great, but here's how to make it even smarter.

#### 1. Implement a "Hard Mode" Penalty
One of the biggest mistakes a Wordle player can make is not using known clues. You can explicitly penalize this.
*   **Action:** In `calculate_total_reward`, after you have calculated `known_green` and `known_yellow`, check if the new `guess` violates "Hard Mode" rules.
    ```python
    # In calculate_total_reward, after calculating known_green/yellow/gray
    hard_mode_penalty = 0.0
    # Check if all known yellow letters are present in the guess
    for letter in known_yellow:
        if letter not in guess:
            hard_mode_penalty += config.reward.get("missing_yellow_penalty", 30.0) # Add this to config
    
    # Check if all known green letters are in their correct spots (your green_violations check already does this)
    # total_penalty += hard_mode_penalty
    ```
*   **Why:** This forces the model to learn the most critical part of deductive reasoning: narrowing down the search space. It stops the model from making "exploratory" guesses in later turns when it should be exploiting known information.

#### 2. Add a "Certainty" or "Knockout" Bonus
When the model correctly deduces that only one possible word remains, it should be heavily rewarded for guessing it.
*   **Action:** In `calculate_total_reward`, before scoring a valid guess, check the number of remaining possibilities.
    ```python
    # In calculate_total_reward, inside the main logic for a valid guess
    clues_before = get_clue_summary([f.guess for f in past_feedback], [f.feedback for f in past_feedback])
    possibilities_before = find_valid_completions(clues_before, constants.ANSWERS_WORDS)
    
    certainty_bonus = 0.0
    if len(possibilities_before) == 1 and guess == possibilities_before[0]:
        certainty_bonus = config.reward.get("certainty_bonus", 50.0) # Add to config
    
    # Add this bonus to the potential_score
    potential_score += certainty_bonus
    ```
*   **Why:** This specifically rewards the "endgame" logic, reinforcing the model's ability to finish the puzzle once the constraints become very tight.

---

### Category 3: Advanced Training & Algorithm Tweaks

These involve more changes to your training loop but can lead to significant stability and performance gains.

#### 1. Periodically Update the Reference Model
In your current `grpo_loss_and_grad`, the `ref_model` is frozen from the start. As the `policy_model` gets better, it can diverge significantly, which can sometimes make training unstable.
*   **Action:** Modify your training loop to periodically update the weights of `ref_model` with a recent version of `policy_model`.
    ```python
    # In the main train() loop
    if step_counter % config.grpo.ref_update_steps == 0: # Add ref_update_steps to config, e.g., 100
        print(f"\n--- Updating reference model at step {step_counter} ---")
        # Get the latest trained LoRA parameters
        latest_params = dict(tree_flatten(policy_model.parameters()))
        # Update the ref_model's weights (it will still be frozen for the loss calculation)
        ref_model.update(tree_unflatten(list(latest_params.items())))
        mx.eval(ref_model.parameters())
    ```
*   **Why:** This keeps the KL-divergence manageable and ensures the GRPO loss is comparing the policy against a more relevant, capable baseline. It can prevent the policy from "forgetting" the base model's capabilities while it learns the new task.

#### 2. Increase `num_generations` in the RL Config
Your GRPO loss learns from preference pairs (winner vs. loser). More pairs give it a better signal.
*   **Action:** In your config, increase `rl.num_generations` from 2 to **4**.
*   **Why:** Generating 4 responses per turn creates up to 3 preference pairs (`(winner, loser1)`, `(winner, loser2)`, `(winner, loser3)`) instead of just one. This provides a much richer and more stable gradient signal for each training step. The trade-off is that each step will take longer.

---

### Category 4: Data and Inference Strategy

#### 1. Curate Data to Focus on "Hard" Games
Your model might be good at games that are solvable in 3-4 turns but struggle with harder ones.
*   **Action:** Analyze your dataset and create a training split that oversamples "hard" trajectories (e.g., those that originally took 5-6 turns to solve).
*   **Why:** This forces the model to train more on the scenarios where it's currently weakest, improving its resilience.

#### 2. Use Self-Consistency at Inference Time
You can improve your evaluation win rate *without retraining* by changing how you generate guesses.
*   **Action:** During evaluation (`play_eval_game`), instead of generating one guess at `temp=0.0`, generate 5 parallel guesses at a low temperature (e.g., `temp=0.2`). Then, choose the guess that appears most frequently (a majority vote).
*   **Why:** This technique, called self-consistency, averages out random generation errors and is known to significantly boost performance on reasoning tasks. The model's most common "reasoned" path is often the correct one.



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


## what are on policy and off policy? Chef analogy


### The Analogy: A Chef Baking Cookies

Imagine a master chef (`the model`) who wants to perfect a cookie recipe (`the policy, π`).

---

#### 1. "The growth in model size... necessitates a large rollout batch size to maximize hardware utilization."

*   **What it means:** Modern AI models run on powerful, expensive GPUs. To use a GPU efficiently, you have to give it a huge amount of work to do all at once. If you only give it a little bit of data, the GPU will sit idle most of the time, wasting time and money.
*   **Analogy:** The chef has a massive, industrial-sized oven (`the GPU`). It's incredibly inefficient to bake just one cookie at a time. To maximize the oven's efficiency, the chef needs to prepare a huge batch of dough for thousands of cookies at once (`a large rollout batch size`).

This first step is the **Data Generation Phase**. The chef (the current model, let's call it `Chef_v1`) bakes 10,000 cookies.

---

#### 2. "To improve sample efficiency... partition a large batch... into multiple mini-batches for gradient updates."

*   **What it means:** While the GPU is great at *generating* data in parallel, the actual *learning* step (calculating the loss and gradients) is memory-intensive. The entire batch of 10,000 cookies (the data) might not fit into the GPU's memory for the learning step. So, we break that big batch into smaller, manageable chunks (`mini-batches`). We will learn from each chunk, one after another.
*   **Analogy:** The chef has all 10,000 cookies ready. Now, they want to taste-test them to learn how to improve the recipe. They can't eat all 10,000 at once. So, they bring a small tray of 64 cookies (`a mini-batch`) to their tasting station (`GPU memory for training`) to analyze.

---

#### 3. "This procedure inevitably introduces an off-policy learning setting..."

This is the most important part.

*   **On-Policy Learning:** The "pure" way to learn. You bake one cookie, taste it, and *immediately* adjust the recipe before baking the next one. The data is always perfectly fresh, generated by the absolute latest version of your recipe. This is slow and doesn't use the industrial oven well.
*   **Off-Policy Learning:** The practical, efficient way. This is what the paragraph describes.

Let's follow the chef's learning process with the mini-batches:

1.  **Taste-test mini-batch #1 (cookies 1-64):** The chef (`Chef_v1`) tastes these cookies. They realize the recipe needs a bit more sugar. They update their recipe book. The chef is now `Chef_v2`.
2.  **Taste-test mini-batch #2 (cookies 65-128):** Now, the chef (`Chef_v2`) tastes the next tray of cookies. **Crucially, these cookies were still baked using the *original* `Chef_v1` recipe.**

This is the "off-policy" problem. The chef (`the current policy πθ`) is learning from data that was generated by a previous, slightly different version of themself (`the old policy πθ_old`). The data is "stale."

---

#### 4. "This also explains the necessity of the clipping mechanism..."

*   **What it means:** As the chef continues this process, their recipe (`πθ`) drifts further and further away from the original `Chef_v1` recipe (`πθ_old`). After 50 mini-batches, the chef might be `Chef_v50`, who uses honey instead of sugar. If `Chef_v50` now tastes a cookie from the original batch (which used sugar), their feedback might be extreme and incorrect ("This tastes nothing like my recipe! It's terrible!"). A massive, incorrect update could ruin the recipe entirely.

*   **The Clipping Mechanism is a safety valve.** It measures how different the current model's opinion is from the old model's opinion on a piece of data (this is the "importance ratio").
    *   If the current model (`Chef_v50`) thinks the old cookie is *way* better or *way* worse than the original model (`Chef_v1`) did, the clipping mechanism steps in.
    *   It says, "Whoa, this data is very 'off-policy' or 'stale'. The difference in opinion is too large. Let's not overreact. We'll limit the size of this learning step to prevent a catastrophic update."

### Summary

In short, the paragraph is explaining this efficient but risky training loop:

1.  **Generate Huge Batch:** Use the current model (`πθ_old`) to generate a massive amount of data to keep the GPU busy.
2.  **Learn in Small Steps:** Break that data into mini-batches to fit in memory for training.
3.  **The "Stale Data" Problem:** After the first mini-batch, the model updates (`πθ`). All subsequent mini-batches from the original data are now "off-policy" (stale).
4.  **The Safety Net:** Use a **clipping mechanism** to prevent the model from making dangerously large updates based on this increasingly stale data.



Let's elaborate on this by clearly separating the process into two distinct phases: the **Generation Phase** and the **Learning Phase**.

---

### Phase 1: The Generation Phase (The "Rollout")

Think of this as the "mass production" step.

1.  **The Goal:** To generate a massive amount of experience (data) as quickly as possible. In our analogy, this is baking the 10,000 cookies.
2.  **The Actor:** The model with its current, fixed set of weights. Let's call this `Model_v1` (`Chef_v1`).
3.  **The Process:**
    *   We give `Model_v1` thousands of different starting prompts (e.g., different Wordle secret words).
    *   The model plays through all these games, generating responses. This is a highly parallel task that GPUs excel at.
    *   **Crucially, during this entire phase, the model is NOT learning.** Its weights are frozen. It's just acting as a data generator. It's like a factory running a full production shift based on a single blueprint.

At the end of Phase 1, we have a massive dataset: a big "bucket" of 10,000 game records, all generated by `Model_v1`.

### Phase 2: The Learning Phase (The "Update")

Now that we have our giant bucket of data, we switch gears to learning.

1.  **The Goal:** To update the model's weights (improve the recipe) using the data we just generated.
2.  **The Process (Step-by-Step Timeline):**
    *   **Mini-Batch #1:** We take the first 64 game records from our bucket. The model (`Model_v1`) analyzes this data, calculates the loss and gradients, and performs an update. **The model's weights have now changed.** It is no longer `Model_v1`; it is now `Model_v2`.
    *   **Mini-Batch #2:** The learning continues. The model needs more data. Where does it get it from? It has two choices:

        *   **Choice A (Slow & Inefficient):** Stop everything. Fire up the expensive Generation Phase again. Have the new `Model_v2` generate a brand new bucket of 10,000 game records. Then take the first 64 from that new bucket to learn from.
        *   **Choice B (Fast & Efficient):** Just grab the *next* 64 game records (games 65-128) from the **original bucket** that we already spent the time and money to create.

    **The system always chooses Choice B.**

This is why `Chef_v2` (or `Model_v2`) still tastes cookies baked by `Chef_v1`. The entire bucket of 10,000 cookies was pre-baked by `Chef_v1` in a single, efficient, mass-production run. The Learning Phase is just the process of working through that pre-existing bucket of data, mini-batch by mini-batch.

### Visualizing the Timeline

| Step | Action | Current Model Version | Data Being Used | Is the Data "Fresh"? |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **GENERATE** 10,000 games | `v1` (Frozen) | (New Data) | N/A |
| **Phase 2** | **LEARN** on Mini-Batch #1 | `v1` -> `v2` | Games 1-64 (from `v1`) | **Yes (On-Policy)** |
| | **LEARN** on Mini-Batch #2 | `v2` -> `v3` | Games 65-128 (from `v1`) | **No (Off-Policy)** |
| | **LEARN** on Mini-Batch #3 | `v3` -> `v4` | Games 129-192 (from `v1`)| **No (Off-Policy)** |
| | ...and so on... | ... | ... | ... |

### The Trade-Off: Speed vs. Freshness

*   **The Inefficient "On-Policy" Way:** Would be to generate a tiny bit of data, learn from it, throw it away, generate new data with the updated model, learn, throw away, and so on. You get perfectly "fresh" data for every single update, but you spend 99% of your time just generating data. This is called being **sample inefficient**.

*   **The Efficient "Off-Policy" Way:** Generate data once in a big, efficient batch. Then, perform many learning steps on that same batch of data. You get far more learning done for the same amount of generation cost. The "price" you pay is that the data becomes progressively more "stale" or "off-policy" as your model's weights change.

This is precisely why the **clipping mechanism** is so essential. It's the safety system that manages the risk of learning from this slightly stale data, allowing us to gain the immense speed and efficiency benefits of this two-phase approach.