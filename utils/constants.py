import re
from utils import read_files


URL_ALLOWED_GUESSES = "https://gist.githubusercontent.com/kcwhite/bb598f1b3017b5477cb818c9b086a5d9/raw/5a0adbbb9830ed93a573cb87a7c14bb5dd0b1883/wordle_possibles.txt"
ALLOWED_GUESSES_PATH = "./data/nyt_possible_wordle_list.txt"

URL_ANSWERS = "https://raw.githubusercontent.com/Roy-Orbison/wordle-guesses-answers/refs/heads/main/answers.txt"
ANSWERS_PATH = "./data/nyt_answers_wordle_list.txt"


# --- Word Lists ---
ALLOWED_GUESSES = read_files.load_word_list_from_url(URL_ALLOWED_GUESSES, ALLOWED_GUESSES_PATH)
ANSWERS_WORDS = list(read_files.load_word_list_from_url(URL_ANSWERS, ANSWERS_PATH))

# --- Matching Patterns ---
# This regex matches a guess inside <guess>...</guess> tags, case-insensitive.
# It captures the guess word, which should be 5 letters long.
GUESS_TAG_RE = re.compile(r"<guess>(.*?)</guess>", re.DOTALL | re.IGNORECASE)
# This should only match 5 ALL-CAPS letters
FIVE_LETTER_WORD_RE = re.compile(r'\b([A-Z]{5})\b')