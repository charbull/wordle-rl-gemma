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

## Generate Synthetic data
```sh
python -m scripts.data_synth --mode rl
```

## Empty your RAM

```sh
sudo pruge
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


### Interesting logs

  -> LoRA model playing...

wordle play: guess 'CORNE' against secret 'BORNE', generated feedback is 'X G G G G'

wordle play: guess 'FORNE' against secret 'BORNE', generated feedback is 'X G G G G'

wordle play: guess 'DORNE' against secret 'BORNE', generated feedback is 'X G G G G'

wordle play: guess 'LORNE' against secret 'BORNE', generated feedback is 'X G G G G'