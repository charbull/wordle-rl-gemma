import os
import requests
from typing import Set


def load_word_list_from_url(url: str, output_path: str) -> Set[str]:
    """
    Loads a set of words from a remote URL.

    Args:
        url (str): The URL of the text file to load.

    Returns:
        Set[str]: A set of uppercase 5-letter words, or an empty set if loading fails.
    """
    try:
        if os.path.exists(output_path):
            print(f"File found reading locally: {output_path}")
            with open(output_path, "r") as f:
                words = {line.strip().upper() for line in f if line.strip()}
            if words:
                return words
                
        # Step 1: Make an HTTP GET request to the URL
        response = requests.get(url)

        # Step 2: Check if the request was successful (status code 200)
        # This will raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Step 3: Process the response text
        # The response text is a single string, so we split it into lines
        words = {
            line.strip().upper()
            for line in response.text.splitlines()
            if len(line.strip()) == 5
        }
        print(f"Successfully loaded {len(words)} words from the URL.")
        if output_path:
            with open(output_path, "w") as f:
                f.write("\n".join(sorted(list(words))))
        return words

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not fetch word list from URL. {e}")
        return set()