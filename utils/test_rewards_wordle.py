import unittest
from unittest.mock import MagicMock, patch

from utils.rewards_wordle import (
    calculate_total_reward,
    GuessFeedback
)

# Mocking the global letter frequencies in case they are used elsewhere, though not in these tests
MOCK_NORMALIZED_FREQS = {
    'E': 1.0, 'A': 0.9, 'R': 0.8, 'I': 0.7, 'O': 0.6,
    'T': 0.5, 'N': 0.4, 'S': 0.3, 'L': 0.2, 'C': 0.1,
}

@patch('utils.rewards_wordle.NORMALIZED_LETTER_FREQS', MOCK_NORMALIZED_FREQS)
class TestCalculateTotalReward(unittest.TestCase):
    def setUp(self):
        """Set up a mock config and a standard game state for all tests."""
        self.mock_config = MagicMock()
        self.mock_config.reward = {
            "format_fail_penalty": -100.0,
            "repetition_penalty": -50.0,
            "solution_correct_guess": 100.0,
            "violated_clue_penalty": -25.0,
            "valid_guess_base": 1.0,
        }
        # A small, clean list of allowed words for testing
        self.allowed_words = {'TRAIN', 'GHOST', 'PLUMB', 'SOLVE', 'CRANE', 'SHAME', 'TABLE'}
        self.secret_word = "SOLVE"

    def test_correct_guess_is_always_winner(self):
        """A correct guess should always receive the highest reward, even if not in the allowed list."""
        response = "<guess>SOLVE</guess>"
        # Note: We are using a different allowed_words set that does NOT contain SOLVE
        # to prove the win condition is checked first.
        custom_allowed_words = {'TRAIN', 'CRANE'}
        reward = calculate_total_reward(response, self.secret_word, [], self.mock_config, custom_allowed_words)
        self.assertEqual(reward, self.mock_config.reward["solution_correct_guess"])

    def test_word_not_in_dictionary(self):
        """Guessing a word not in the allowed list should receive the format_fail_penalty."""
        response = "<guess>APPLE</guess>" # APPLE is not in self.allowed_words
        reward = calculate_total_reward(response, self.secret_word, [], self.mock_config, self.allowed_words)
        self.assertEqual(reward, self.mock_config.reward["format_fail_penalty"])

    def test_repeated_guess(self):
        """Repeating a previous guess should result in a specific penalty."""
        past_feedback = [GuessFeedback(guess="TRAIN", feedback="x x x x x")]
        response = "<guess>TRAIN</guess>"
        reward = calculate_total_reward(response, self.secret_word, past_feedback, self.mock_config, self.allowed_words)
        self.assertEqual(reward, self.mock_config.reward["repetition_penalty"])

    def test_violation_of_gray_letter(self):
        """Using a known gray letter should result in the violated_clue_penalty."""
        # In CRANE vs SOLVE, C, R, A, N are gray.
        past_feedback = [GuessFeedback(guess="CRANE", feedback="x x x x ✓")]
        # GHOST uses 'O' (yellow), 'S' (green), but also 'C' which is gray from CRANE.
        # This test is tricky, let's simplify.
        # Let's say secret is TABLE. First guess is SONIC -> S,O,N,I are gray.
        secret_word = "TABLE"
        past_feedback = [GuessFeedback(guess="SONIC", feedback="x x x x x")]
        # Now guess CRANE, which contains gray 'C' and 'N'.
        response = "<guess>CRANE</guess>"
        reward = calculate_total_reward(response, secret_word, past_feedback, self.mock_config, self.allowed_words)
        self.assertEqual(reward, self.mock_config.reward["violated_clue_penalty"])

    def test_violation_of_green_letter_position(self):
        """Failing to use a known green letter in its correct spot should be penalized."""
        # In SOLVE, if 'S' is green in pos 0, a guess not starting with 'S' is a violation.
        past_feedback = [GuessFeedback(guess="SHAME", feedback="✓ x x x ✓")] # S is green
        response = "<guess>GHOST</guess>" # Does not start with S
        reward = calculate_total_reward(response, self.secret_word, past_feedback, self.mock_config, self.allowed_words)
        self.assertEqual(reward, self.mock_config.reward["violated_clue_penalty"])

    def test_violation_of_yellow_letter_usage(self):
        """Failing to use a known yellow letter anywhere in the guess should be penalized."""
        # In SOLVE, if 'L' is yellow, a guess without 'L' is a violation.
        past_feedback = [GuessFeedback(guess="PLUMB", feedback="x ✓ x x x")] # L is yellow
        response = "<guess>GHOST</guess>" # Does not contain L
        reward = calculate_total_reward(response, self.secret_word, past_feedback, self.mock_config, self.allowed_words)
        self.assertEqual(reward, self.mock_config.reward["violated_clue_penalty"])

    def test_valid_first_guess(self):
        """A valid first guess that follows no rules (as there are none) gets the base reward."""
        response = "<guess>TRAIN</guess>"
        expected_total_reward = self.mock_config.reward["valid_guess_base"]
        reward = calculate_total_reward(response, self.secret_word, [], self.mock_config, self.allowed_words)
        self.assertAlmostEqual(reward, expected_total_reward)

    def test_valid_guess_that_follows_all_clues(self):
        """A valid guess that correctly uses all green and yellow clues should get the base reward."""
        secret_word = "TABLE"
        # From CRANE vs TABLE -> A is yellow, E is yellow. C, R, N are gray.
        past_feedback = [GuessFeedback(guess="CRANE", feedback="x x - x -")]
        # SHAME uses A and E, and no gray letters C, R, N. This is a valid move.
        response = "<guess>SHAME</guess>"
        expected_total_reward = self.mock_config.reward["valid_guess_base"]
        reward = calculate_total_reward(response, secret_word, past_feedback, self.mock_config, self.allowed_words)
        self.assertAlmostEqual(reward, expected_total_reward)


if __name__ == '__main__':
    unittest.main(verbosity=2)