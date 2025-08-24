import unittest
import json
from unittest.mock import patch, MagicMock
import io
import src.synth.cot_wordle_data_generation as wdg

class TestWordleGame(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('src.wordle.game.get_feedback')
        self.mock_get_feedback = self.patcher.start()
        
        def feedback_side_effect(guess, secret_word):
            if guess == "APPLE" and secret_word == "apple":
                return MagicMock(feedback="G G G G G")
            if guess == "GRAPE" and secret_word == "apple":
                return MagicMock(feedback="X X Y Y G")
            return MagicMock(feedback="X X X X X")
        
        self.mock_get_feedback.side_effect = feedback_side_effect
        self.game = wdg.WordleGame("APPLE", ["APPLE", "GRAPE", "LEMON", "PEACH"], max_turns=3)

    def tearDown(self):
        self.patcher.stop()

    def test_make_guess_correct(self):
        self.game.make_guess("APPLE")
        self.assertTrue(self.game.is_over)
        self.assertIn("APPLE", self.game.guesses)
        self.assertIn("G G G G G", self.game.feedback)

    def test_make_guess_incorrect(self):
        self.game.make_guess("GRAPE")
        self.assertFalse(self.game.is_over)
        self.assertIn("GRAPE", self.game.guesses)
        self.assertIn("X X Y Y G", self.game.feedback)

class TestHelperFunctions(unittest.TestCase):
    
    def test_get_clue_summary(self):
        # This test now confirms that the fix in get_clue_summary works correctly.
        # Secret Word could be TRAIN
        guesses = ["CRANE", "TRASH"]
        # Feedback for CRANE vs TRAIN -> X G G Y X
        # Feedback for TRASH vs TRAIN -> G G G X X
        feedback = ["X G G Y X", "G G G X X"]
        
        clues = wdg.get_clue_summary(guesses, feedback)
        
        self.assertEqual(clues["greens"], ['T', 'R', 'A', '_', '_'])
        self.assertEqual(clues["yellows"], {'N'})
        self.assertEqual(clues["greys"], {'C', 'E', 'S', 'H'})
        self.assertEqual(clues["yellow_positions"]['N'], {3})

    @patch('src.wordle.game.get_feedback')
    def test_find_best_guess_entropy(self, mock_get_feedback):
        possible_words = ["BEAST", "FEAST", "LEAST"]
        allowed_guesses = ["FEAST", "BROIL"]
        
        def mock_feedback_func(guess, secret_word):
            result = MagicMock()
            result.guess = guess

            if guess == "BROIL":
                # This mock data splits possibilities into 3 unique groups for max entropy.
                if secret_word == "BEAST":
                    result.feedback = "Y X X X X"
                elif secret_word == "FEAST":
                    result.feedback = "X X X X X"
                elif secret_word == "LEAST":
                    result.feedback = "X X Y X X"
            elif guess == "FEAST":
                # This guess splits possibilities into 2 groups (sizes 2 and 1).
                if secret_word == "BEAST" or secret_word == "LEAST":
                    result.feedback = "X G G G G"
                elif secret_word == "FEAST":
                    result.feedback = "G G G G G"
            
            return result        
        mock_get_feedback.side_effect = mock_feedback_func
        best_guess = wdg.find_best_guess(possible_words, allowed_guesses)
        self.assertEqual(best_guess, "BROIL")




class TestDataGenerationPipeline(unittest.TestCase):

    @patch('src.synth.cot_wordle_data_generation.find_best_guess')
    @patch('random.choice')
    @patch('random.randint')
    def test_generate_rl_data_point_with_history(self, mock_randint, mock_choice, mock_find_best_guess):
        """
        Tests that generate_cot_wordle_data produces a correctly formatted JSON object.
        """
        solution_words = ["GRAZE", "SOARE", "GAINS", "OTHER", "GIANT"] 
        mock_randint.return_value = 3
        secret_word = "GIANT"
        mock_choice.side_effect = [secret_word, "GRAZE", "GAINS"]
        mock_find_best_guess.return_value = "OTHER"
        # --- 2. Run the Function and Capture Output ---
        string_buffer = io.StringIO()
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            mock_file.return_value.write = string_buffer.write

            wdg.generate_cot_rl_data(
                num_samples=1,
                output_file="dummy_rl_data.jsonl",
                prompt="Test System Prompt",
                solution_words=solution_words
            )

        output_string = string_buffer.getvalue()
        self.assertTrue(len(output_string) > 0, "Data generation produced empty output.")
        
        try:
            data = json.loads(output_string)['data']
        except (json.JSONDecodeError, KeyError):
            self.fail("Data generation did not write a valid JSON string with the expected structure.")

        self.assertEqual(data['secret'], secret_word)
        
        messages = data['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        
        user_prompt = messages[1]['content']
        
        # Game state after guess 1 "SLATE" (G G X Y G) and guess 2 "CRATE" (X Y X Y G) vs "STYLE"
        self.assertIn("**Correct Position (Green):** `G _ A N _", user_prompt)
        self.assertIn("**Wrong Position (Yellow):** 'I' (at least 1), 'T' (at least 1)", user_prompt)
        self.assertIn("**Not in Word (Gray):** E, H, O, R, S, Z", user_prompt)
        self.assertIn("**Words Already Guessed:** OTHER, SOARE, GRAZE, GAINS", user_prompt)

        
    @patch('src.synth.cot_wordle_data_generation.find_best_guess')
    @patch('random.choice')
    @patch('random.randint')
    def test_generate_rl_data_point_no_history(self, mock_randint, mock_choice, mock_find_best_guess):
        """Tests the case where num_previous_turns is 0."""

        solution_words = ["GRAZE", "SOARE", "GAINS", "OTHER", "GIANT"] 
        mock_randint.return_value = 0 # Force a game with no history
        secret_word = "OTHER"
        mock_choice.return_value = secret_word

        mock_find_best_guess.return_value = "SOARE"
        string_buffer = io.StringIO()
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            mock_file.return_value.write = string_buffer.write
            test_prompt = "Test System Prompt"
            wdg.generate_cot_rl_data(
                num_samples=1, 
                output_file="dummy_path.jsonl", 
                prompt=test_prompt,
                solution_words=solution_words
            )

        output_string = string_buffer.getvalue()
        data = json.loads(output_string)['data']
        user_prompt = data['messages'][1]['content']

        self.assertEqual(data['secret'], secret_word)
        
        messages = data['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        
        user_prompt = messages[1]['content']

        # Check that it's the standard "first turn" prompt
        self.assertIn("**Correct Position (Green):** `_ _ _ _ _`", user_prompt)
        self.assertIn("Words Already Guessed:** SOARE", user_prompt)
        self.assertIn("**Not in Word (Gray):** A, S\n", user_prompt)
        self.assertIn("Wrong Position (Yellow):** 'E' (at least 1), 'O' (at least 1), 'R' (at least 1)\n* ", user_prompt)
          

    
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)