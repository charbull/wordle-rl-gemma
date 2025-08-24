import json
import math
import numpy as np
import mlx.core as mx
from collections import Counter
from tqdm import tqdm
import time
from typing import Set
from src.utils import constants
# =====================================================================
# VECTORIZED FEEDBACK & ENTROPY CALCULATION
# =====================================================================

def words_to_mlx_array(words: Set[str]) -> mx.array:
    """Converts a set of 5-letter words to an MLX integer array."""
    char_to_int = {char: i for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    # Sort the words to ensure a deterministic order in the array
    sorted_words = sorted(list(words))
    int_lists = [[char_to_int[char] for char in word] for word in sorted_words]
    return mx.array(int_lists, dtype=mx.uint8)


def get_feedback_vectorized(guess_vec: mx.array, solutions_arr: mx.array) -> np.ndarray:
    """
    Calculates feedback for one guess against all solutions using MLX arrays.
    '✓' = 2 (Green), '-' = 1 (Yellow), 'x' = 0 (Gray)
    """
    greens_mx = (guess_vec == solutions_arr)
    
    guess_np = np.array(guess_vec, copy=False)[0]
    solutions_np = np.array(solutions_arr, copy=False)
    greens_np = np.array(greens_mx, copy=False)

    patterns = np.zeros_like(solutions_np, dtype=np.int8)
    patterns[greens_np] = 2

    for i in range(solutions_np.shape[0]):
        sol_counts = Counter(solutions_np[i, ~greens_np[i,:]])
        
        for j in range(5):
            if not greens_np[i, j]:
                letter = guess_np[j]
                if sol_counts[letter] > 0:
                    patterns[i, j] = 1
                    sol_counts[letter] -= 1
    
    return patterns


def calculate_shannon_entropy(probabilities: list) -> float:
    """Calculates the Shannon entropy for a list of probabilities."""
    return sum(-p * math.log2(p) for p in probabilities if p > 0)


# =====================================================================
# 3. MAIN SCRIPT
# =====================================================================

if __name__ == "__main__":
    print("Starting MLX-accelerated pre-computation of Wordle information gain...")
    start_time = time.time()

    # --- CONFIGURATION ---
    solution_words = constants.ANSWERS_WORDS
    allowed_guesses = constants.ALLOWED_GUESSES
   
    # Print a few words to confirm it worked
    if solution_words:
        print("\nPossible word list: ", len(solution_words))
        print("Sample of possible answers words:")
        print(list(solution_words)[:2])

    if allowed_guesses:
        print("\nFull allowed guesses list: ", len(allowed_guesses))
        print("Sample of full allowed guesses words:")
        print(list(allowed_guesses)[:2])    

    # Local paths to save/load the files
    OUTPUT_FILE = './data/word_entropy.json'

   
    if not solution_words or not allowed_guesses:
        print("\nCould not load required word lists. Exiting.")
        exit()

    # The full list of guesses to test should include all possible solutions
    allowed_guesses.update(solution_words)

    # --- CONVERSION TO MLX ARRAYS ---
    print("\n--- Converting words to MLX arrays ---")
    solutions_arr = words_to_mlx_array(solution_words)
    
    num_solutions = len(solution_words)
    word_entropy_scores = {}

    print(f"\n--- Starting Entropy Calculation for {len(allowed_guesses)} guesses against {num_solutions} solutions ---")
    for guess_word in tqdm(sorted(list(allowed_guesses)), desc="Calculating Entropy"):
        guess_vec = words_to_mlx_array({guess_word})
        patterns_arr = get_feedback_vectorized(guess_vec, solutions_arr)
        
        _, counts = np.unique(patterns_arr, axis=0, return_counts=True)
        
        probabilities = counts / num_solutions
        entropy = calculate_shannon_entropy(probabilities)
        word_entropy_scores[guess_word] = entropy

    # --- SAVING ---
    if word_entropy_scores:
        print(f"\n--- Calculation Complete. Saving scores to {OUTPUT_FILE} ---")
        sorted_scores = dict(sorted(word_entropy_scores.items(), key=lambda item: item[1], reverse=True))

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(sorted_scores, f, indent=2)

        print("\n--- Top 20 Best Starting Guesses (by Information Gain) ---")
        for i, (word, score) in enumerate(list(sorted_scores.items())[:20]):
            print(f"{i+1:2d}. {word}: {score:.4f} bits")
        
        end_time = time.time()
        print(f"\n✅ Successfully saved entropy scores.")
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    else:
        print("\nNo entropy scores were calculated.")