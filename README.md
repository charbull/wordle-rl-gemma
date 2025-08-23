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