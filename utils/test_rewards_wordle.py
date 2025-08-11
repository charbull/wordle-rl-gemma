import unittest
from unittest.mock import MagicMock

from utils.rewards_wordle import (
    format_prompt_from_dataset,
    parse_guess,
    calculate_total_reward,
    GuessFeedback
)

class TestFormatPromptFromDataset(unittest.TestCase):
    """Tests for the format_prompt_from_dataset function."""

    def test_first_turn_empty_history(self):
        """Should return the first-turn prompt for an empty history string."""
        sample = {'past_guess_history': '[]'}
        expected_prompt = "This is the first turn. Please provide your best starting word."
        self.assertEqual(format_prompt_from_dataset(sample), expected_prompt)

    def test_first_turn_malformed_history(self):
        """Should return the first-turn prompt if history is malformed."""
        sample = {'past_guess_history': 'not a valid list'}
        expected_prompt = "This is the first turn. Please provide your best starting word."
        self.assertEqual(format_prompt_from_dataset(sample), expected_prompt)
        
    def test_single_past_guess(self):
        """Should correctly format a prompt with one previous guess."""
        history_str = "[('ARISE', 'x ✓ x - x')]"
        sample = {'past_guess_history': history_str}
        expected_prompt = "**Clues so far:**\n* Guess 1: ARISE → x ✓ x - x"
        self.assertEqual(format_prompt_from_dataset(sample), expected_prompt)

    def test_multiple_past_guesses(self):
        """Should correctly format a prompt with multiple previous guesses."""
        history_str = "[('ARISE', 'x ✓ x - x'), ('TRULY', '✓ - x x -')]"
        sample = {'past_guess_history': history_str}
        expected_prompt = (
            "**Clues so far:**\n"
            "* Guess 1: ARISE → x ✓ x - x\n"
            "* Guess 2: TRULY → ✓ - x x -"
        )
        self.assertEqual(format_prompt_from_dataset(sample), expected_prompt)
        
    def test_missing_history_key(self):
        """Should raise a KeyError if 'past_guess_history' is missing."""
        sample = {'other_data': 123}
        with self.assertRaises(KeyError):
            format_prompt_from_dataset(sample)


class TestParseGuess(unittest.TestCase):
    """Tests for the parse_guess function."""

    def test_simple_case(self):
        """Should extract a simple guess in uppercase."""
        response = "<think>I will guess a word.</think><guess>TRAIN</guess>"
        self.assertEqual(parse_guess(response), "TRAIN")

    def test_uppercase_with_whitespace(self):
        """Should handle a valid uppercase guess with extra whitespace."""
        response = "<guess>  CRANE  </guess>"
        self.assertEqual(parse_guess(response), "CRANE")

    def test_with_punctuation(self):
        """Should ignore punctuation inside the tags."""
        response = "<guess>My guess is: 'QUERY'!</guess>"
        self.assertEqual(parse_guess(response), "QUERY")

    def test_no_guess_tag(self):
        """Should return None if <guess> tag is missing."""
        response = "<think>I'm thinking...</think>"
        self.assertIsNone(parse_guess(response))
        
    def test_no_5_letter_word_in_tag(self):
        """Should return None if no 5-letter word is found inside the tag."""
        response = "<guess>I am stuck.</guess>"
        self.assertIsNone(parse_guess(response))

    def test_multiple_5_letter_words(self):
        """Should return the LAST 5-letter word found."""
        response = "<guess>First I thought HOUSE, but maybe FABLE is better.</guess>"
        self.assertEqual(parse_guess(response), "FABLE")

    def test_malformed_tag(self):
        """Should return None if tags are not properly closed."""
        response = "<guess>AUDIO"
        self.assertIsNone(parse_guess(response))

    def test_case_insensitive_tags(self):
        """Should find the guess even with varied tag casing."""
        # The payload MUST be uppercase, but the tags can be any case.
        response = "<THINK>...</THINK><GUESS>REACT</gUeSs>"
        self.assertEqual(parse_guess(response), "REACT")

    def test_six_letter_word(self):
        """Should ignore words that are not exactly 5 letters long."""
        response = "<guess>Maybe GHOSTS is the word?</guess>"
        self.assertIsNone(parse_guess(response))


class TestCalculateTotalReward(unittest.TestCase):
    """Tests for the calculate_total_reward function."""

    def setUp(self):
        """Set up a mock config and allowed words for all tests."""
        # Create a mock config object that simulates the real config structure
        self.mock_config = MagicMock()
        self.mock_config.reward = {
            "format_fail_penalty": -100.0,
            "repetition_penalty": -50.0,
            "solution_correct_guess": 100.0,
            "violated_clue_penalty": -25.0,
            "valid_guess_base": 1.0,
        }
        self.allowed_words = {'TRAIN', 'GHOST', 'PLUMB', 'SOLVE', 'CRANE'}
        self.secret_word = "SOLVE"

    def test_correct_guess(self):
        """Should return the max reward for a correct guess."""
        response = "<guess>SOLVE</guess>"
        reward = calculate_total_reward(
            response, self.secret_word, [], self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["solution_correct_guess"])

    def test_invalid_format_no_guess(self):
        """Should return a large penalty for a malformed response."""
        response = "<think>I don't know.</think>"
        reward = calculate_total_reward(
            response, self.secret_word, [], self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["format_fail_penalty"])
        
    def test_guess_not_in_word_list(self):
        """Should return a large penalty for a guess not in the allowed list."""
        response = "<guess>XXXXX</guess>"
        reward = calculate_total_reward(
            response, self.secret_word, [], self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["format_fail_penalty"])

    def test_repeated_guess(self):
        """Should return the repetition penalty for a repeated guess."""
        response = "<guess>TRAIN</guess>"
        past_feedback = [GuessFeedback(guess="TRAIN", feedback="x x x x x")]
        reward = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["repetition_penalty"])

    def test_valid_first_guess(self):
        """Should return the base reward for a valid, non-violating first guess."""
        response = "<guess>TRAIN</guess>"
        reward = calculate_total_reward(
            response, self.secret_word, [], self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["valid_guess_base"])

    def test_violation_of_gray_letter(self):
        """Should penalize using a known gray letter."""
        response = "<guess>GHOST</guess>" # 'G' and 'H' are gray
        # Secret is SOLVE. Past guess was CRANE -> C,R,A,N are gray. E is yellow.
        past_feedback = [GuessFeedback(guess="CRANE", feedback="x x x x -")]
        reward = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["violated_clue_penalty"])

    def test_violation_of_green_letter(self):
        """Should penalize not using a green letter in its correct spot."""
        response = "<guess>GHOST</guess>" # Does not have 'L' at index 2
        # Secret is SOLVE. Past guess was PLUMB -> L is green at index 2
        past_feedback = [GuessFeedback(guess="PLUMB", feedback="x ✓ x x x")]
        reward = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["violated_clue_penalty"])
        
    def test_violation_of_yellow_letter(self):
        """Should penalize not including a known yellow letter in the guess."""
        response = "<guess>GHOST</guess>" # Does not contain 'L'
        # Secret is SOLVE. Past guess was PLUMB -> L is yellow. Let's adjust.
        # Secret: 'PROXY'. Past Guess: 'SOLVE' -> O is yellow, L is gray.
        # past_feedback: [GuessFeedback('SOLVE', '- x x x x')]
        secret_word = "PROXY"
        past_feedback = [GuessFeedback(guess="SOLVE", feedback="- x x x x")]
        reward = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["violated_clue_penalty"])

    def test_valid_guess_following_clues(self):
        """Should return base reward for a valid guess that respects all clues."""
        # --- SCENARIO SETUP ---
        # Secret Word: TABLE
        # Guess 1: CRANE
        # Correct Feedback for CRANE vs TABLE: C(x) R(x) A(✓) N(x) E(✓)
        secret_word = "TABLE"
        past_feedback = [GuessFeedback(guess="CRANE", feedback="x x ✓ x ✓")]
        
        # From this, the model should know:
        # - Green: A at index 2, E at index 4
        # - Gray: C, R, N
        
        # --- TEST GUESS ---
        # The guess "SHAME" is a valid word that follows all the rules:
        # - It does not use the gray letters C, R, N.
        # - It correctly places the green letter A at index 2.
        # - It correctly places the green letter E at index 4.
        # - It is NOT the solution.
        self.allowed_words.add("SHAME")
        response = "<guess>SHAME</guess>"
        
        reward = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config, self.allowed_words
        )
        self.assertEqual(reward, self.mock_config.reward["valid_guess_base"])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)