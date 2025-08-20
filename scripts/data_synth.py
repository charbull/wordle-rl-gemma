# This script generates synthetic Wordle data for training models.
# It can generate data for either Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL) modes.
# The generated data is saved in JSONL format.
import argparse
from utils import read_files
from utils import prompt
from synth.cot_wordle_data_generation import generate_cot_sft_data, generate_cot_rl_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Wordle data for SFT or RL training.")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['sft', 'rl'], 
        help="The type of data to generate: 'sft' for Supervised Fine-Tuning or 'rl' for Reinforcement Learning."
    )
    args = parser.parse_args()

    NUM_SAMPLES_TO_GENERATE = 2300 # this is the number of answers of the NYT Wordle game
    OUTPUT_FILENAME = f"./data/wordle_cot_data_{args.mode}.jsonl" # Dynamic filename
    file_url_possible_answers = "https://raw.githubusercontent.com/Roy-Orbison/wordle-guesses-answers/refs/heads/main/answers.txt"
    SOLUTION_WORDS = list(read_files.load_word_list_from_url(file_url_possible_answers, "./data/nyt_answers_wordle_list.txt"))

    if args.mode == 'sft':
        generate_cot_sft_data(
            num_samples=NUM_SAMPLES_TO_GENERATE,
            output_file=OUTPUT_FILENAME,
            prompt=prompt.SYSTEM_PROMPT,
            solutions_words=SOLUTION_WORDS
        )
    elif args.mode == 'rl':
        generate_cot_rl_data(
            num_samples=NUM_SAMPLES_TO_GENERATE,
            output_file=OUTPUT_FILENAME,
            prompt=prompt.SYSTEM_PROMPT,
            solution_words=SOLUTION_WORDS
        )