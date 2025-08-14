import unittest
from unittest.mock import MagicMock, patch

# Make sure the import path matches your project structure
from utils.rewards_wordle import (
    calculate_total_reward,
    GuessFeedback,
    # We may need to mock this function, so let's make sure it's importable
    # If it's in the same file, this is fine.
)

class TestCalculateTotalReward(unittest.TestCase):
    def setUp(self):
        """Set up a mock config, tokenizer, and standard game state for all tests."""
        self.mock_config = MagicMock()
        # REVISED: Updated with new graduated penalties and bonuses
        self.mock_config.reward = {
            "format_fail_penalty": -100.0,
            "repetition_penalty": -75.0,
            "not_in_dictionary_penalty": -100.0,
            "solution_correct_guess": 150.0,
            "valid_guess_base": 5.0,
            # Graduated Penalties
            "gray_letter_penalty": -5.0,
            "yellow_letter_penalty": -10.0,
            "green_position_penalty": -15.0,
            # Behavioral Penalties/Bonuses
            "time_penalty_per_guess": -1.0,
            "length_penalty_per_token": -0.01,
            "entropy_unique_letter_bonus": 5.0,
            "entropy_common_letter_bonus": 5.0,
        }
        
        # Mock the tokenizer for deterministic length penalties
        self.mock_tokenizer = MagicMock()
        # Let's say every response is encoded to 20 tokens for simplicity in testing
        self.mock_tokenizer.encode.return_value = [0] * 20

        self.allowed_words = {'TRAIN', 'GHOST', 'PLUMB', 'SOLVE', 'CRANE', 'SHAME', 'TABLE'}
        self.secret_word = "SOLVE"

    def test_correct_guess_is_always_winner(self):
        """A correct guess's game_score should be the solution_correct_guess reward."""
        response = "<guess>SOLVE</guess>"
        game_score, _ = calculate_total_reward(
            response, self.secret_word, [], self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertEqual(game_score, self.mock_config.reward["solution_correct_guess"])

    def test_word_not_in_dictionary(self):
        """Guessing a word not in the allowed list should have a format_fail_penalty game_score."""
        response = "<guess>APPLE</guess>"
        game_score, _ = calculate_total_reward(
            response, self.secret_word, [], self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertEqual(game_score, self.mock_config.reward["format_fail_penalty"])

    def test_repeated_guess(self):
        """Repeating a guess should have a repetition_penalty game_score."""
        past_feedback = [GuessFeedback(guess="TRAIN", feedback="x x x x x")]
        response = "<guess>TRAIN</guess>"
        game_score, _ = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertEqual(game_score, self.mock_config.reward["repetition_penalty"])

    def test_violation_of_gray_letters(self):
        """Using known gray letters should result in a sum of gray_letter_penalties."""
        secret_word = "TABLE"
        # From SONIC vs TABLE, letters S, O, N, I, C are gray.
        past_feedback = [GuessFeedback(guess="SONIC", feedback="x x x x x")]
        # CRANE contains two gray letters: C and N.
        response = "<guess>CRANE</guess>"
        
        expected_penalty = 2 * self.mock_config.reward["gray_letter_penalty"] # 2 * -5.0 = -10.0
        
        game_score, _ = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertEqual(game_score, expected_penalty)

    def test_violation_of_green_letter_position(self):
        """Failing to use a known green letter in its spot should be penalized."""
        past_feedback = [GuessFeedback(guess="SHAME", feedback="✓ x x x ✓")] # Greens: S at 0, E at 4. Grays: H, A, M
        response = "<guess>GHOST</guess>" # Violates 2 greens (S,E) and 1 gray (H)
        
        game_score, _ = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        
        expected_penalty = (2 * self.mock_config.reward["green_position_penalty"]) + \
                        (1 * self.mock_config.reward["gray_letter_penalty"]) # (2*-15) + (1*-5) = -35.0
        self.assertEqual(game_score, expected_penalty)


    def test_violation_of_yellow_letter_usage(self):
        """Failing to use a known yellow letter anywhere in the guess should be penalized."""
        past_feedback = [GuessFeedback(guess="PLUMB", feedback="x - x x x")] # L is yellow. P,U,M,B are gray.
        response = "<guess>GHOST</guess>"
        
        game_score, _ = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        
        self.assertEqual(game_score, self.mock_config.reward["yellow_letter_penalty"])

    @patch('utils.rewards_wordle.reward_for_entropy_proxy', return_value=10.0)
    def test_valid_guess_that_follows_all_clues(self, mock_entropy_proxy):
        """A valid guess following all clues should get the base reward + entropy bonus."""
        secret_word = "TABLE"
        # From CRANE vs TABLE -> A is yellow, E is yellow. C, R, N are gray.
        past_feedback = [GuessFeedback(guess="CRANE", feedback="x x - x -")]
        # SHAME uses A and E, and no gray letters C, R, N. This is a valid move.
        response = "<guess>SHAME</guess>"
        
        expected_game_score = self.mock_config.reward["valid_guess_base"] + 10.0 # 5.0 + 10.0 = 15.0
        
        game_score, _ = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertAlmostEqual(game_score, expected_game_score)

    @patch('utils.rewards_wordle.reward_for_entropy_proxy', return_value=10.0)
    def test_training_reward_includes_behavioral_penalties(self, mock_entropy_proxy):
        """The final training_reward should include game_score and behavioral penalties."""
        response = "<guess>TRAIN</guess>" # A valid first guess
        
        # 1. Calculate expected game_score
        expected_game_score = self.mock_config.reward["valid_guess_base"] + 10.0 # 5.0 + 10.0 = 15.0
        
        # 2. Calculate expected behavioral penalties
        time_penalty = self.mock_config.reward["time_penalty_per_guess"] # -1.0
        # self.mock_tokenizer returns a list of 20 tokens
        length_penalty = 20 * self.mock_config.reward["length_penalty_per_token"] # 20 * -0.01 = -0.2
        
        # 3. Calculate final expected training_reward
        expected_training_reward = expected_game_score + time_penalty + length_penalty # 15.0 - 1.0 - 0.2 = 13.8

        _, training_reward = calculate_total_reward(
            response, self.secret_word, [], self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertAlmostEqual(training_reward, expected_training_reward)

if __name__ == '__main__':
    unittest.main(verbosity=2)