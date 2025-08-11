import unittest
from unittest.mock import MagicMock, patch

from utils.rewards_wordle import (
    calculate_total_reward,
    reward_for_entropy_proxy,
    GuessFeedback
)

# Mocking the global letter frequencies for consistent testing
MOCK_NORMALIZED_FREQS = {
    'E': 1.0, 'A': 0.9, 'R': 0.8, 'I': 0.7, 'O': 0.6,
    'T': 0.5, 'N': 0.4, 'S': 0.3, 'L': 0.2, 'C': 0.1,
    'Z': 0.01
}

# Helper function to avoid repeating calculation logic in tests
def calculate_expected_entropy(guess, config):
    unique_letters = set(guess)
    unique_bonus = (len(unique_letters) / 5.0) * config.reward.get("entropy_unique_letter_bonus")
    common_letter_score = sum(MOCK_NORMALIZED_FREQS.get(letter, 0) for letter in unique_letters)
    common_bonus = (common_letter_score / 3.5) * config.reward.get("entropy_common_letter_bonus")
    return unique_bonus + common_bonus

@patch('utils.rewards_wordle.NORMALIZED_LETTER_FREQS', MOCK_NORMALIZED_FREQS)
class TestRewardForEntropyProxy(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.reward = {
            "entropy_unique_letter_bonus": 2.0,
            "entropy_common_letter_bonus": 3.0,
        }

    def test_high_entropy_guess(self):
        guess = "ARISE"
        expected_reward = calculate_expected_entropy(guess, self.mock_config)
        reward = reward_for_entropy_proxy(guess, self.mock_config)
        self.assertAlmostEqual(reward, expected_reward)

    def test_low_entropy_guess_repeated_letters(self):
        """Should give a lower reward for a guess with repeated, common letters."""
        # This test is now clearer and uses letters from our mock list.
        guess = "EERIE"
        expected_reward = calculate_expected_entropy(guess, self.mock_config)
        reward = reward_for_entropy_proxy(guess, self.mock_config)
        self.assertAlmostEqual(reward, expected_reward)

    def test_low_entropy_guess_uncommon_letters(self):
        guess = "JAZZY" # J and Y are not in MOCK_NORMALIZED_FREQS
        expected_reward = calculate_expected_entropy(guess, self.mock_config)
        reward = reward_for_entropy_proxy(guess, self.mock_config)
        self.assertAlmostEqual(reward, expected_reward)

    def test_invalid_guess(self):
        self.assertEqual(reward_for_entropy_proxy("", self.mock_config), 0.0)

@patch('utils.rewards_wordle.NORMALIZED_LETTER_FREQS', MOCK_NORMALIZED_FREQS)
class TestCalculateTotalReward(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.reward = {
            "format_fail_penalty": -100.0,
            "repetition_penalty": -50.0,
            "solution_correct_guess": 100.0,
            "violated_clue_penalty": -25.0,
            "valid_guess_base": 1.0,
            "entropy_unique_letter_bonus": 2.0,
            "entropy_common_letter_bonus": 3.0,
        }
        self.allowed_words = {'TRAIN', 'GHOST', 'PLUMB', 'SOLVE', 'CRANE', 'SHAME', 'TABLE'}
        self.secret_word = "SOLVE"

    def test_correct_guess(self):
        response = "<guess>SOLVE</guess>"
        reward = calculate_total_reward(response, self.secret_word, [], self.mock_config, self.allowed_words)
        self.assertEqual(reward, self.mock_config.reward["solution_correct_guess"])
    
    def test_violation_of_gray_letter(self):
        response = "<guess>GHOST</guess>"
        past_feedback = [GuessFeedback(guess="CRANE", feedback="x x x x -")]
        reward = calculate_total_reward(response, self.secret_word, past_feedback, self.mock_config, self.allowed_words)
        self.assertEqual(reward, self.mock_config.reward["violated_clue_penalty"])

    def test_valid_first_guess_with_entropy_bonus(self):
        response = "<guess>TRAIN</guess>"
        expected_entropy_bonus = calculate_expected_entropy("TRAIN", self.mock_config)
        expected_total_reward = self.mock_config.reward["valid_guess_base"] + expected_entropy_bonus

        reward = calculate_total_reward(response, self.secret_word, [], self.mock_config, self.allowed_words)
        self.assertAlmostEqual(reward, expected_total_reward)

    def test_valid_guess_following_clues_with_entropy_bonus(self):
        secret_word = "TABLE"
        past_feedback = [GuessFeedback(guess="CRANE", feedback="x x ✓ x ✓")]
        response = "<guess>SHAME</guess>"
        
        expected_entropy_bonus = calculate_expected_entropy("SHAME", self.mock_config)
        expected_total_reward = self.mock_config.reward["valid_guess_base"] + expected_entropy_bonus
        
        reward = calculate_total_reward(response, secret_word, past_feedback, self.mock_config, self.allowed_words)
        self.assertAlmostEqual(reward, expected_total_reward)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load tests from each specific test class
    suite.addTests(loader.loadTestsFromTestCase(TestRewardForEntropyProxy))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateTotalReward))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)