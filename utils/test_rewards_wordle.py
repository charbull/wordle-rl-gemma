import unittest
from unittest.mock import MagicMock, patch
from collections import Counter

from utils.rewards_wordle import (
    calculate_total_reward,
    GuessFeedback,
    get_feedback,
    calculate_stagnation_penalty,
    format_prompt_for_model
)

# A fake dictionary of entropy scores to be used by our mock.
# This simulates the presence of the word_entropy.json file.
# In test_rewards.py
FAKE_ENTROPY_SCORES = {
    "CRANE": 5.88,
    "STARS": 5.70,
    "CLOTS": 5.65,
    "PLUMB": 4.50,
    "TRAIN": 5.80
}

@patch('utils.rewards_wordle.WORD_ENTROPY_SCORES', FAKE_ENTROPY_SCORES)
class TestCalculateTotalReward(unittest.TestCase):
    def setUp(self):
        """Set up a mock config, tokenizer, and standard game state for all tests."""
        self.mock_config = MagicMock()
        self.mock_config.reward = {
            "solution_correct_guess": 150.0,
            "valid_guess_base": 10.0,
            
            # --- New Hybrid Strategic Bonuses ---
            "information_gain_bonus_coeff": 2.0, # For Turn 1
            "new_letter_bonus": 1.5,              # For each new letter on Turns > 1

            # --- Behavioral Penalties ---
            "length_penalty_per_token": 0.01,
            "time_penalty_per_guess": 1.0,

            # --- Clue Violation Penalties ---
            "gray_letter_penalty": 15.0,
            "yellow_letter_penalty": 15.0,
            "green_position_penalty": 20.0,

            "green_reuse_penalty": 0.5,
            "yellow_reuse_penalty": 0.2,
            
            # --- Game Rule Violation Penalties ---
            "repetition_penalty": 30.0,
            "not_in_dictionary_penalty": 25.0,
            "format_fail_penalty": 120.0
        }
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode.return_value = [0] * 20 # Assume 20 tokens per response
        
        self.allowed_words = {
                    'TRAIN', 'GHOST', 'PLUMB', 'SOLVE', 'CRANE', 'SHAME', 'TABLE',
                    'SONIC', 'STARS', 'CLOTS', 'STIFF', 'DOREN', 'ABOVE',
                    'CLONE', 'STONE', 'TONER', 'STORE', 'DRONE'
                }
        self.secret_word = "SOLVE"

    # =====================================================================
    # --- Tests for Game Rule Violations ---
    # =====================================================================

    def test_correct_guess_is_winner(self):
        """A correct guess's score should be the `solution_correct_guess` reward."""
        game_score, _ = calculate_total_reward(
            "<guess>SOLVE</guess>", self.secret_word, [], self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertEqual(game_score, self.mock_config.reward["solution_correct_guess"])

    def test_repeated_guess_gets_penalty(self):
        """Repeating a guess should return the `repetition_penalty`."""
        past_feedback = [GuessFeedback(guess="TRAIN", feedback="X X X X X")]
        expected_score = -self.mock_config.reward["repetition_penalty"]
        game_score, _ = calculate_total_reward(
            "<guess>TRAIN</guess>", self.secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertEqual(game_score, expected_score)

    def test_no_parsable_guess_gets_penalty(self):
        """A response with no valid guess tag should return the `format_fail_penalty`."""
        expected_score = -self.mock_config.reward["format_fail_penalty"]
        game_score, _ = calculate_total_reward(
            "I think the word is SOLVE.", self.secret_word, [], self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertEqual(game_score, expected_score)

    # =====================================================================
    # --- Tests for Hybrid Strategic Bonuses ---
    # =====================================================================

    def test_turn1_uses_information_gain_bonus(self):
        """On Turn 1, the strategic bonus should come from the pre-calculated entropy scores."""
        past_feedback = [] # This makes it Turn 1
        response = "<guess>CRANE</guess>"

        # Expected score = base + (entropy * coeff)
        info_gain = FAKE_ENTROPY_SCORES["CRANE"] * self.mock_config.reward["information_gain_bonus_coeff"]
        expected_score = self.mock_config.reward["valid_guess_base"] + info_gain
        # 10.0 + (5.88 * 2.0) = 21.76
        
        game_score, _ = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertAlmostEqual(game_score, expected_score)

    def test_turn2_uses_new_letter_exploration_bonus(self):
        """On Turn 2+, the strategic bonus should be for introducing new, unseen letters."""
        # On Turn 1, we guessed 'CRANE'. C,R,A,N,E are now "known letters".
        past_feedback = [GuessFeedback(guess="CRANE", feedback="X X X X X")]
        response = "<guess>PLUMB</guess>" # 'PLUMB' introduces 5 entirely new letters.

        # Expected score = base + (num_new_letters * bonus_per_letter)
        expected_bonus = 5 * self.mock_config.reward["new_letter_bonus"]
        expected_score = self.mock_config.reward["valid_guess_base"] + expected_bonus
        # 10.0 + (5 * 1.5) = 17.5
        
        game_score, _ = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertAlmostEqual(game_score, expected_score)

    def test_turn2_bonus_is_correct_for_mixed_new_and_old_letters(self):
        """REVISED: Tests the exploration bonus with a clear, unambiguous state."""
        secret_word = "GHOST"
        # Turn 1: Guessed 'TRAIN'. Feedback vs GHOST is all gray.
        # Known letters are {T,R,A,I,N}. All are gray.
        past_feedback = [GuessFeedback(guess="TRAIN", feedback="X X X X X")]
        
        # Turn 2: Guess 'SHAME'. S,H,M,E are new. A is old and gray.
        response = "<guess>SHAME</guess>"
        
        # 1. Calculate the bonus
        # 4 new letters (S,H,M,E) * 1.5 bonus/letter = 6.0
        expected_bonus = 4 * self.mock_config.reward["new_letter_bonus"]
        potential_score = self.mock_config.reward["valid_guess_base"] + expected_bonus
        # 10.0 + 6.0 = 16.0
        
        # 2. Calculate penalties
        # The guess 'SHAME' uses 'A', which is a known gray letter.
        gray_penalty = 1 * self.mock_config.reward["gray_letter_penalty"] # 15.0
        
        # 3. Final Score
        expected_score = potential_score - gray_penalty # 16.0 - 15.0 = 1.0
        
        game_score, _ = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config,
            self.allowed_words, self.mock_tokenizer
        )
        self.assertAlmostEqual(game_score, expected_score)

    # =====================================================================
    # --- Tests for Combined Penalties ---
    # =====================================================================

    def test_out_of_dictionary_word_gets_penalty_and_no_bonus(self):
        """An OOD word should get a penalty and a zero strategic bonus."""
        past_feedback = [] # Turn 1
        self.allowed_words = {'CRANE'} # Ensure 'GIZMO' is not in the list
        response = "<guess>GIZMO</guess>"

        # Strategic bonus is 0 because OOD words aren't in the entropy list.
        potential_score = self.mock_config.reward["valid_guess_base"]
        ood_penalty = self.mock_config.reward["not_in_dictionary_penalty"]
        expected_score = potential_score - ood_penalty # 10.0 - 25.0 = -15.0
        
        game_score, _ = calculate_total_reward(
            response, self.secret_word, past_feedback, self.mock_config, self.allowed_words, self.mock_tokenizer
        )
        self.assertAlmostEqual(game_score, expected_score)

    def test_full_score_with_clue_violations_and_bonuses(self):
        """Tests the final calculation with both a bonus and multiple clue penalties."""
        secret_word = "GHOST"
        # Turn 1 was 'CRANE', C,R,A,N,E are gray.
        past_feedback = [GuessFeedback(guess="CRANE", feedback="X X X X X")]
        # Guess 'PLUMB' introduces 5 new letters.
        # But 'PLUMB' vs 'GHOST' has feedback 'x x x x x', so no clue violations.
        
        # Let's try a different scenario.
        secret_word = "TABLE"
        # Turn 1: 'SONIC'. S,O,N,I,C are gray.
        past_feedback = [GuessFeedback(guess="SONIC", feedback="X X X X X")]
        # Turn 2: Guess 'PLUMB'. 5 new letters. Violates gray 'C' from 'SONIC'. No.
        # Guess 'SHAME'. S is new, H new, A new, M new, E new. S is gray.
        # Okay, let's craft one.
        secret_word = "SOLVE"
        # Turn 1: 'TRAIN'. T,R,A,I,N are gray.
        past_feedback = [GuessFeedback(guess="TRAIN", feedback="X X X X X")]
        # Turn 2: 'GHOST'. G,H,O,S,T are new letters.
        # GHOST vs SOLVE -> S- O✓ L- V- E-. No, S- O✓ x x x.
        # So S is yellow, O is green.
        past_feedback = [GuessFeedback(guess="GHOST", feedback="Y G X X X")]
        response = "<guess>PLUMB</guess>" # P, L, U, M, B are 5 new letters.
                                          # Fails to use Yellow 'S'. Fails to use Green 'O'.
        
        # 1. Calculate the bonus
        bonus = 5 * self.mock_config.reward["new_letter_bonus"] # 5 * 1.5 = 7.5
        potential_score = self.mock_config.reward["valid_guess_base"] + bonus # 10.0 + 7.5 = 17.5
        
        # 2. Calculate penalties
        yellow_penalty = 1 * self.mock_config.reward["yellow_letter_penalty"] # 15.0 for missing S
        green_penalty = 1 * self.mock_config.reward["green_position_penalty"] # 20.0 for missing O
        total_penalty = yellow_penalty + green_penalty # 15.0 + 20.0 = 35.0
        
        # 3. Final Score
        expected_score = potential_score - total_penalty # 17.5 - 35.0 = -17.5
        
        game_score, _ = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config,
            self.allowed_words, self.mock_tokenizer
        )
        self.assertAlmostEqual(game_score, expected_score)

# =====================================================================
# --- REGRESSION TESTS (Safeguards against old bugs) ---
# =====================================================================

    def test_stars_vs_clots_is_correct_with_info_gain(self):
        """REGRESSION TEST: Ensures info gain correctly prefers STARS over CLOTS on Turn 1."""
        secret_word = "STIFF"
        past_feedback = [] # Turn 1
        
        # With a correct `is_valid_guess`, STARS should not be penalized.
        info_gain_stars = FAKE_ENTROPY_SCORES["STARS"] * self.mock_config.reward["information_gain_bonus_coeff"]
        expected_score_stars = self.mock_config.reward["valid_guess_base"] + info_gain_stars
        
        info_gain_clots = FAKE_ENTROPY_SCORES["CLOTS"] * self.mock_config.reward["information_gain_bonus_coeff"]
        expected_score_clots = self.mock_config.reward["valid_guess_base"] + info_gain_clots
        
        score_stars, _ = calculate_total_reward(
            "<guess>STARS</guess>", secret_word, past_feedback, self.mock_config, 
            self.allowed_words, self.mock_tokenizer
        )
        score_clots, _ = calculate_total_reward(
            "<guess>CLOTS</guess>", secret_word, past_feedback, self.mock_config, 
            self.allowed_words, self.mock_tokenizer
        )
        
        self.assertAlmostEqual(score_stars, expected_score_stars)
        self.assertAlmostEqual(score_clots, expected_score_clots)
        self.assertGreater(score_stars, score_clots)



    def test_clue_cleanup_promotes_yellow_to_green(self):
        """
        REGRESSION TEST: A simple, precise test to ensure a letter that becomes
        green is correctly removed from the yellow and gray sets.
        This version has the CORRECTED expected score calculation.
        """
        secret_word = "STYLE"
        self.allowed_words.update({"SLATE", "TEPAL"})

        # Turn 1: Guess 'SLATE'. Feedback vs STYLE: S✓ L✓ A x T- E✓
        # This establishes the clue state:
        # - Green: S at 0, L at 1, E at 4
        # - Yellow: T
        # - Gray: A
        past_feedback = [
            GuessFeedback(guess="SLATE", feedback="G G X Y G")
        ]
        
        # Turn 2: Guess 'TEPAL'.
        response = "<guess>TEPAL</guess>"
        
        # --- Calculate Expected Score ---

        # 1. Bonus (Turn 2)
        # Known letters from SLATE: {S,L,A,T,E}. New letters in TEPAL: {P}. (1 new letter)
        bonus = 1 * self.mock_config.reward["new_letter_bonus"]
        potential_score = self.mock_config.reward["valid_guess_base"] + bonus # 10 + 1.5 = 11.5
        
        # 2. Penalties
        # Correct final clues after SLATE: Green:{0:'S', 1:'L', 4:'E'}, Yellow:{'T'}, Gray:{'A'}
        # Guess 'TEPAL':
        # - Green violations: Violates S(0)->T, L(1)->E, E(4)->L. (3 violations)
        # - Yellow violations: Uses T. (0 violations)
        # - Gray violations: Uses known gray letter 'A'. (1 violation)
        
        green_penalty = 3 * self.mock_config.reward["green_position_penalty"] # 3 * 20.0 = 60.0
        gray_penalty = 1 * self.mock_config.reward["gray_letter_penalty"]     # 1 * 15.0 = 15.0
        total_penalty = green_penalty + gray_penalty                         # 75.0
        stagnation_penalty = self.mock_config.reward["yellow_reuse_penalty"] # 0.2 for reusing T which is yellow
        # 3. Final Score
        expected_score = potential_score - total_penalty - stagnation_penalty # 11.5 - 75.0 - 0.2 = -63.5

        game_score, _ = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config,
            self.allowed_words, self.mock_tokenizer
        )
        
        self.assertAlmostEqual(game_score, expected_score)

    def test_game_state_from_stage_log(self):
        """
        REGRESSION TEST: Precisely recreates the game state from the debug log
        for the secret word 'STAGE' to validate both clue aggregation and
        violation counting.
        """
        # 1. --- SETUP THE EXACT SCENARIO ---
        secret_word = "STAGE"
        self.allowed_words.add("STAGE")
        self.allowed_words.add("GAVEL")
        self.allowed_words.add("CHASE")

        # The game history after Turn 1
        # Feedback for 'GAVEL' vs 'STAGE' is G(-), A(✓), V(x), E(✓), L(x)
        past_feedback = [
            GuessFeedback(guess="GAVEL", feedback="Y G X G X")
        ]
        
        # The guess to be evaluated in Turn 2
        response = "<guess>CHASE</guess>"
        
        # --- EXPECTED STATE AND CALCULATIONS ---
        # Correctly aggregated clues after 'GAVEL':
        # known_green = {1: 'A', 3: 'E'}
        # known_yellow = {'G'}
        # known_gray = {'V', 'L'}
        
        # 2. Calculate the expected strategic bonus (for Turn 2)
        # Known letters from GAVEL: {G,A,V,E,L}. New letters in CHASE: {C,H,S}. (3 new)
        bonus = 3 * self.mock_config.reward["new_letter_bonus"]
        potential_score = self.mock_config.reward["valid_guess_base"] + bonus # 10.0 + 4.5 = 14.5

        # 3. Calculate expected penalties for 'CHASE' against the correct clue state
        # - Green violations: Fails to place 'A' at index 1 and 'E' at index 3. (2 violations)
        # - Yellow violations: Fails to use the known yellow letter 'G'. (1 violation)
        # - Gray violations: Does not use 'V' or 'L'. (0 violations)
        
        green_penalty = 2 * self.mock_config.reward["green_position_penalty"] # 2 * 20.0 = 40.0
        yellow_penalty = 1 * self.mock_config.reward["yellow_letter_penalty"] # 1 * 15.0 = 15.0
        total_penalty = green_penalty + yellow_penalty                       # 55.0

        # 4. Calculate the final expected game score
        expected_score = potential_score - total_penalty # 14.5 - 55.0 = -40.5
        
        # --- EXECUTE AND ASSERT ---
        game_score, _ = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config,
            self.allowed_words, self.mock_tokenizer
        )
        
        self.assertAlmostEqual(game_score, expected_score)

    def test_game_state_from_stage_log_2(self):
        """
        REGRESSION TEST: Precisely recreates the game state from the debug log
        for the secret word 'STAGE' to validate both clue aggregation and
        violation counting.
        
        This test will fail if green clues ('✓') are incorrectly added to the
        yellow set, or if green violations are not counted correctly.
        """
        # 1. --- SETUP THE EXACT SCENARIO FROM THE LOG ---
        secret_word = "STAGE"
        # Ensure all relevant words are in the allowed dictionary for the test
        self.allowed_words.update({"STAGE", "GAVEL", "CHASE"})

        # The game history after Turn 1.
        # Feedback for 'GAVEL' vs 'STAGE' is G(-), A(✓), V(x), E(✓), L(x)
        past_feedback = [
            GuessFeedback(guess="GAVEL", feedback="Y G X G X")
        ]
        
        # The guess to be evaluated in Turn 2
        response = "<guess>CHASE</guess>"
        
        # --- EXPECTED STATE AND CALCULATIONS ---
        # After 'GAVEL', the correct clue state should be:
        # known_green = {1: 'A', 3: 'E'}
        # known_yellow = {'G'}
        # known_gray = {'V', 'L'}
        
        # 2. Calculate the expected strategic bonus (for Turn 2)
        # Known letters from GAVEL: {G,A,V,E,L}. New letters in CHASE: {C,H,S}. (3 new)
        bonus = 3 * self.mock_config.reward["new_letter_bonus"]
        potential_score = self.mock_config.reward["valid_guess_base"] + bonus # 10.0 + 4.5 = 14.5

        # 3. Calculate expected penalties for 'CHASE' against the CORRECT clue state
        # - Green violations: Fails to place 'A' at index 1 and 'E' at index 3. (2 violations)
        # - Yellow violations: Fails to use the known yellow letter 'G'. (1 violation)
        # - Gray violations: Does not use 'V' or 'L'. (0 violations)
        
        green_penalty = 2 * self.mock_config.reward["green_position_penalty"] # 2 * 20.0 = 40.0
        yellow_penalty = 1 * self.mock_config.reward["yellow_letter_penalty"] # 1 * 15.0 = 15.0
        total_penalty = green_penalty + yellow_penalty                       # 55.0

        # 4. Calculate the final expected game score
        expected_score = potential_score - total_penalty # 14.5 - 55.0 = -40.5
        
        # --- EXECUTE AND ASSERT ---
        # To properly test this, we need to see the intermediate values.
        # We will modify the function under test to return them, a common TDD practice.
        # For now, let's assume the function is NOT modified and just check the final score.
        
        game_score, _ = calculate_total_reward(
            response, secret_word, past_feedback, self.mock_config,
            self.allowed_words, self.mock_tokenizer
        )
        
        self.assertAlmostEqual(game_score, expected_score,
            msg="Final game score is incorrect, indicating a bug in either clue aggregation or violation counting.")


    def test_get_feedback_for_stage_scenario(self):
            """
            Tests the get_feedback function directly for the 'STAGE' scenario.
            This provides a clear and isolated failure if the function is not correct.
            """
            # 1. --- SETUP ---
            secret_word = "STAGE"
            guess = "GAVEL"
            
            # 2. --- DEFINE EXPECTED RESULT ---
            # This is the known, correct feedback string for the inputs above.
            expected_feedback = "Y Y X Y X"
            
            # 3. --- EXECUTE THE FUNCTION ---
            # Call the function directly to get its actual output.
            actual_feedback_obj = get_feedback(guess, secret_word)
            
            # 4. --- ASSERT AND PROVIDE A CLEAR MESSAGE ---
            # Check if the actual output matches the expected output.
            self.assertEqual(actual_feedback_obj.feedback, expected_feedback,
                msg="\n\n>>> The 'get_feedback' function in 'rewards_wordle.py' is still incorrect. Please replace it with the correct version. <<<")

class TestStagnationPenalty(unittest.TestCase):
    def setUp(self):
        """Set up a mock config object for all stagnation tests."""
        self.mock_config = MagicMock()
        self.mock_config.reward = {
            "green_reuse_penalty": 0.5,
            "yellow_reuse_penalty": 0.2,
            # Add other necessary rewards for the integration test
            "valid_guess_base": 10.0,
            "new_letter_bonus": 1.5,
            "green_position_penalty": 20.0,
            "yellow_letter_penalty": 15.0,
            "gray_letter_penalty": 15.0,
            "not_in_dictionary_penalty": 25.0
        }
        self.allowed_words = {"FRESH", "BRICK", "BRAVE", "GHOST", "STOMP", "SASSY", "COOTS", "ROOTS"}

    def test_no_stagnation_for_fresh_guess(self):
        """Should return 0.0 penalty when no letters are known."""
        known_green = {}
        known_yellow = Counter()
        guess = "FRESH"
        penalty = calculate_stagnation_penalty(guess, known_green, known_yellow, self.mock_config)
        self.assertEqual(penalty, 0.0)

    def test_penalty_for_multiple_green_reuse(self):
        """Should sum penalties for multiple reused green letters."""
        known_green = {1: 'R', 4: 'E'}
        known_yellow = Counter()
        guess = "BRAVE"
        penalty = calculate_stagnation_penalty(guess, known_green, known_yellow, self.mock_config)
        self.assertAlmostEqual(penalty, 1.0, msg="Penalty should be 0.5 for 'R' + 0.5 for 'E'")

    def test_penalty_for_multiple_yellow_reuse(self):
        """Should sum penalties for multiple unique reused yellow letters."""
        known_green = {}
        known_yellow = Counter({'S': 1, 'T': 2})
        guess = "STOMP"
        penalty = calculate_stagnation_penalty(guess, known_green, known_yellow, self.mock_config)
        self.assertAlmostEqual(penalty, 0.4, msg="Penalty should be 0.2 for 'S' + 0.2 for 'T'")

    def test_penalty_for_duplicate_yellow_in_guess(self):
        """Should penalize a reused yellow letter only once per guess."""
        known_green = {}
        known_yellow = Counter({'S': 1})
        guess = "SASSY"
        penalty = calculate_stagnation_penalty(guess, known_green, known_yellow, self.mock_config)
        self.assertAlmostEqual(penalty, 0.2, msg="Penalty for 'S' should only be counted once.")

    def test_combined_penalty_for_coots_scenario(self):
        """Tests the exact scenario from the logs: 'ROOTS' -> 'COOTS'."""
        known_green = {1: 'O', 2: 'O', 3: 'T'}
        known_yellow = Counter({'S': 1})
        guess = "COOTS"
        penalty = calculate_stagnation_penalty(guess, known_green, known_yellow, self.mock_config)
        # 0.5 for O at pos 1 + 0.5 for O at pos 2 + 0.5 for T at pos 3 = 1.5 (green)
        # 0.2 for S = 0.2 (yellow)
        # Total = 1.7
        self.assertAlmostEqual(penalty, 1.7)

    def test_total_reward_integration_for_coots_scenario(self):
        """
        Integration test verifying that the stagnation penalty correctly reduces the
        final game score for a strategically poor guess.
        """
        secret_word = "SOOTH"
        # Simulate the game state after guessing 'ROOTS'
        past_feedback = [GuessFeedback(guess='ROOTS', feedback='X G G G Y')]
        
        # This is the stagnant guess that should be penalized
        response_with_stagnant_guess = "<guess>COOTS</guess>"
        
        # Calculate the score.
        game_score, _ = calculate_total_reward(
            response=response_with_stagnant_guess,
            secret_word=secret_word,
            past_feedback=past_feedback,
            config=self.mock_config,
            allowed_words=self.allowed_words,
            tokenizer=None # Tokenizer not needed for this calculation
        )
        
        # --- Manually calculate the expected score ---
        # 1. Bonus: Turn 2. Known letters from 'ROOTS' are {R,O,T,S}.
        # Guess 'COOTS' introduces one new letter 'C'.
        bonus = 1 * self.mock_config.reward["new_letter_bonus"]
        potential_score = self.mock_config.reward["valid_guess_base"] + bonus # 10.0 + 1.5 = 11.5
        
        # 2. Penalties:
        # Clue Violations for 'COOTS' are 0.
        # Stagnation penalty is 1.7 (0.5*3 for greens, 0.2 for yellow 'S').
        total_penalty = 1.7
        
        # 3. Final Score
        expected_game_score = potential_score - total_penalty # 11.5 - 1.7 = 9.8
        
        self.assertAlmostEqual(game_score, expected_game_score)



class TestFormatPromptForModel(unittest.TestCase):

    def setUp(self):
        """Set up a mock system prompt for all tests."""
        self.system_prompt = "You are a Wordle solving assistant."

    def test_first_turn_prompt_is_correct(self):
        """Should generate the special introductory prompt for the first turn."""
        past_feedback = []
        messages = format_prompt_for_model(past_feedback, self.system_prompt)
        
        expected_content = "This is the first turn. Please provide your best starting word."
        self.assertEqual(messages[-1]['role'], 'user')
        self.assertEqual(messages[-1]['content'], expected_content)


    def test_prompt_with_only_yellows_and_grays(self):
        """Should correctly list yellow letters and gray letters."""
        past_feedback = [
            GuessFeedback(guess="RAISE", feedback="Y Y X Y X")
        ]
        messages = format_prompt_for_model(past_feedback, self.system_prompt)
        user_content = messages[-1]['content']

        expected_lines = [
            "**Current Knowledge:**",
            "*   **Green Letters (Correct Position):** `_ _ _ _ _`",
            "*   **Yellow Letters (In word, wrong position):** '\'A\'' (at least 1), '\'R\'' (at least 1), '\'S\'' (at least 1)",
            "*   **Gray Letters (Not in word):** E, I",
            "\nBased on this summary, what is your next guess?"
        ]
        expected_content = "\n".join(expected_lines).replace('\'\'', '\'')

        self.assertEqual(user_content, expected_content)

    def test_prompt_with_duplicate_yellow_counts(self):
        """
        Tests the function's interpretation of a complex feedback string.
        Note: The input feedback "Y X Y X X" for "EERIE" is unusual, but this test
        validates that the function correctly translates exactly what it's given.
        """
        past_feedback = [
            GuessFeedback(guess="EERIE", feedback="Y X Y X X")
        ]
        messages = format_prompt_for_model(past_feedback, self.system_prompt)
        user_content = messages[-1]['content']

        # --- FIX: Update expected content to match the function's correct logical output ---
        expected_lines = [
            "**Current Knowledge:**",
            "*   **Green Letters (Correct Position):** `_ _ _ _ _`",
            # The function correctly identifies 'E' (once) and 'R' as yellow from the feedback.
            "*   **Yellow Letters (In word, wrong position):** '\'E\'' (at least 1), '\'R\'' (at least 1)",
            # The function correctly identifies 'I' as gray, but not 'R' or the other 'E's.
            "*   **Gray Letters (Not in word):** I",
            "\nBased on this summary, what is your next guess?"
        ]
        expected_content = "\n".join(expected_lines).replace('\'\'', '\'')
        
        self.assertEqual(user_content, expected_content)

    def test_complex_prompt_from_piper_scenario(self):
        """Tests the prompt generation from a multi-turn game state."""
        # This simulates the state after guessing CREWS and then PLIER for the secret PIPER
        past_feedback = [
            GuessFeedback(guess="CREWS", feedback="X Y Y X X"), # R, E are yellow. C,W,S are gray
            GuessFeedback(guess="PLIER", feedback="G X Y G G")  # P,E,R are green. I is yellow. L is gray.
        ]
        messages = format_prompt_for_model(past_feedback, self.system_prompt)
        user_content = messages[-1]['content']

        # The state logic should correctly deduce:
        # Green: P _ _ E R
        # Yellow: I (R and E were promoted to green)
        # Gray: C, W, S, L
        expected_lines = [
            "**Current Knowledge:**",
            "*   **Green Letters (Correct Position):** `P _ _ E R`",
            "*   **Yellow Letters (In word, wrong position):** '\'I\'' (at least 1)",
            "*   **Gray Letters (Not in word):** C, L, S, W", # Alphabetically sorted
            "\nBased on this summary, what is your next guess?"
        ]
        expected_content = "\n".join(expected_lines).replace('\'\'', '\'')

        self.assertEqual(user_content, expected_content)


if __name__ == '__main__':
    unittest.main(verbosity=2)