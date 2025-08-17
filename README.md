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
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Calculate Enthropy offline
```sh
python3 calculate_word_entropy_mlx.py
```



## Run training
```sh
python3 train_gemma_rl.py 
```


## Run unit tests
```sh
python3 -m unittest
```