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

### Evaluation

* when Chose from Allowed words in the dictionary, there is more words than the 2900 list of answers that the model was trained on.
* during training and sampling evaluation, the eval data had a random (0, 4) history turns where we picked carefully logical attempts that brings the model closer to the solution.


## Design choices

* less strict, give feedback on words that are not in the possible words for wordle
* instead of just providing the rewards functions, I decided to play a whole game of wordle at each step
* generate 2 plays of wordle at each step, pick the best guess each time to advance the state of the game.

## Open questions

* how far can I go with a 3B model, is it possible to get the upper theoritcal limit without actually training for days/weeks. Scaling laws: https://arxiv.org/abs/2001.08361
* 