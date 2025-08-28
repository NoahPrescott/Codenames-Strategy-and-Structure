### Code for collecting GPT relatedness judgments and Codenames guesses. 

import os
import openai
import json
import time
import re
from typing import Dict, List
from copy import deepcopy

# Configuration
ADDITIONAL_RELATEDNESS_SCORES = 4
TESTING_MODE = False  # Set to False for full run
TEST_BOARDS_COUNT = 2
TEST_RELATEDNESS_COUNT = 3
API_DELAY = 0.5
TEMPERATURE = 1.0
MAX_RETRIES = 5

# Prompts 
QUERY_OPENER = "I am going to give you a one-word clue, along with a list of 12 words. I chose the clue to help you guess exactly 3 of the words in the list. Your task is to list the 3 words that you think I have in mind based on the clue."
QUERY_CLOSER = "Please simply list the 3 words that you think I have in mind based on the clue."
RELATEDNESS_OPENER = "I am going to provide you with 2 words. Your task is to assess them on a relatedness scale of 1 to 100, 100 being very related and 0 being not related."
RELATEDNESS_CLOSER = "Please just provide your score."

def get_response(query: str) -> str:
    """Get GPT response with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": query}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise Exception(f"Failed after {MAX_RETRIES} attempts")

def create_guess_query(clue: str, words: List[str]) -> str:
    """Create guessing query"""
    return f"{QUERY_OPENER}\nClue: {clue.lower()}\nWords: {', '.join([w.lower() for w in words])}\n{QUERY_CLOSER}"

def create_relatedness_query(clue: str, word: str) -> str:
    """Create relatedness query"""
    return f"{RELATEDNESS_OPENER}\nWords: {word.lower()}, {clue.lower()}\n{RELATEDNESS_CLOSER}"

def parse_guess_response(response: str, board_words: List[str]) -> List[str]:
    """Parse GPT guess response"""
    print(f"    Raw response: '{response}'")
    
    # Clean response
    response = response.strip()
    response = re.sub(r'^["\'\(\[\{]*|["\'\)\]\}]*$', '', response)
    
    # Method 1: Comma-separated
    if ',' in response:
        match = re.search(r'([A-Z][A-Z]+)\s*,\s*([A-Z][A-Z]+)\s*,\s*([A-Z][A-Z]+)', response.upper())
        if match:
            words = [match.group(1), match.group(2), match.group(3)]
            if all(w in board_words for w in words):
                print(f"    Parsed: {words}")
                return words
    
    # Method 2: Find any valid board words and take first 3
    found_words = [w for w in board_words if w in response.upper()]
    if len(found_words) >= 3:
        result = found_words[:3]
        print(f"    Parsed: {result}")
        return result
    
    print(f"    WARNING: Could not parse. Found: {found_words}")
    return found_words + ["UNKNOWN"] * (3 - len(found_words))

def parse_relatedness_response(response: str) -> float:
    """Parse relatedness score"""
    numbers = re.findall(r'\d+\.?\d*', response)
    if numbers:
        score = max(0, min(100, float(numbers[0])))
        print(f"    Parsed score: {score}")
        return score
    print(f"    WARNING: No score found in '{response}'")
    return 0.0

def expand_gpt_guesses(boards_data: Dict) -> Dict:
    """Expand GPT guesses to match human sample sizes"""
    print("=== EXPANDING GPT GUESSES ===")
    data = deepcopy(boards_data)
    items = list(boards_data.items())[:TEST_BOARDS_COUNT] if TESTING_MODE else boards_data.items()
    
    for i, (key, board) in enumerate(items):
        print(f"\nBoard {i+1}/{len(items)}: {key[:50]}...")
        words = board["words"]
        
        # GPT clue condition
        if "gpt_clue" in board and "human_guess_gpt_clue" in board:
            needed = len(board["human_guess_gpt_clue"]) - 1
            if needed > 0:
                print(f"  Getting {needed} more GPT guesses for GPT clue '{board['gpt_clue']}'")
                guesses = [board["gpt_guess_gpt_clue"]] if isinstance(board["gpt_guess_gpt_clue"], list) else []
                
                for j in range(needed):
                    query = create_guess_query(board["gpt_clue"], words)
                    if TESTING_MODE: print(f"    Query: {query}")
                    response = get_response(query)
                    guess = parse_guess_response(response, words)
                    guesses.append(guess)
                    time.sleep(API_DELAY)
                
                data[key]["gpt_guess_gpt_clue"] = guesses
        
        # Human clue condition  
        if "human_clue" in board and "human_guess_human_clue" in board:
            needed = len(board["human_guess_human_clue"]) - 1
            if needed > 0:
                print(f"  Getting {needed} more GPT guesses for human clue '{board['human_clue']}'")
                guesses = [board["gpt_guess_human_clue"]] if isinstance(board["gpt_guess_human_clue"], list) else []
                
                for j in range(needed):
                    query = create_guess_query(board["human_clue"], words)
                    if TESTING_MODE: print(f"    Query: {query}")
                    response = get_response(query)
                    guess = parse_guess_response(response, words)
                    guesses.append(guess)
                    time.sleep(API_DELAY)
                
                data[key]["gpt_guess_human_clue"] = guesses
    
    return data

def expand_gpt_relatedness(relatedness_data: List[Dict]) -> List[Dict]:
    """Expand GPT relatedness scores"""
    print("=== EXPANDING GPT RELATEDNESS ===")
    data = deepcopy(relatedness_data)
    gpt_indices = [i for i, entry in enumerate(data) if entry.get("source") == "gpt"]
    
    if TESTING_MODE:
        gpt_indices = gpt_indices[:TEST_RELATEDNESS_COUNT]
    
    print(f"Expanding {len(gpt_indices)} GPT relatedness entries")
    
    for i, idx in enumerate(gpt_indices):
        entry = data[idx]
        clue, word = entry["clue"], entry["word"]
        print(f"\nEntry {i+1}/{len(gpt_indices)}: '{clue}' + '{word}'")
        
        # Start with existing score
        scores = [entry["relatedness"]] if isinstance(entry["relatedness"], (int, float)) else entry["relatedness"].copy()
        
        # Get additional scores
        for j in range(ADDITIONAL_RELATEDNESS_SCORES):
            query = create_relatedness_query(clue, word)
            if TESTING_MODE: print(f"    Query: {query}")
            response = get_response(query)
            score = parse_relatedness_response(response)
            scores.append(score)
            time.sleep(API_DELAY)
        
        data[idx]["relatedness"] = scores
        print(f"  Final scores: {scores}")
    
    return data

def main():
    """Main execution"""
    print("GPT Data Collection Script")
    print("=" * 40)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Set OPENAI_API_KEY environment variable")
        return
    
    # Load data
    try:
        with open('boards-data.json') as f:
            boards_data = json.load(f)
        with open('relatedness-data.json') as f:
            relatedness_data = json.load(f)
        print(f"Loaded {len(boards_data)} boards, {len(relatedness_data)} relatedness entries")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Expand data
    try:
        expanded_boards = expand_gpt_guesses(boards_data)
        expanded_relatedness = expand_gpt_relatedness(relatedness_data)
        
        # Save results
        with open('boards-data-expanded.json', 'w') as f:
            json.dump(expanded_boards, f, indent=2)
        with open('relatedness-data-expanded.json', 'w') as f:
            json.dump(expanded_relatedness, f, indent=2)
        
        print(f"\n{'='*40}")
        print("SUCCESS! Saved expanded data files.")
        print(f"Testing mode: {TESTING_MODE}")
        if TESTING_MODE:
            print("Set TESTING_MODE = False for full run")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    