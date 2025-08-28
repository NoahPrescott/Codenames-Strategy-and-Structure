import json
import numpy as np
from collections import defaultdict

def count_exp_1(data):
    ages = []
    total_females = 0
    total_males = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        # age
        age = entry.get("age")
        if isinstance(age, int) or (isinstance(age, str) and age.isdigit()):
            ages.append(int(age))
        # gender
        gender = entry.get("gender", "").strip().lower()
        if gender == "female":
            total_females += 1
        elif gender == "male":
            total_males += 1
    total_other = len(data) - total_females - total_males
    # Age stats
    if ages:
        median_age = np.median(ages)
        mean_age = np.mean(ages)
        std_age = np.std(ages, ddof=1)
        min_age = np.min(ages)
        max_age = np.max(ages)
    else:
        median_age = mean_age = std_age = min_age = max_age = None
    return {
        "total_participants": len(data),
        "total_females": total_females,
        "total_males": total_males,
        "total_other": total_other,
        "median_age": median_age,
        "mean_age": mean_age,
        "std_age": std_age,
        "min_age": min_age,
        "max_age": max_age
    }

def count_exp_2(data):
    ages = []
    total_females = 0
    total_males = 0
    excluded = []
    participants = data["__collections__"]["exptDataHumanClue"]
    for pid, pdata in participants.items():
        trials = pdata.get("trialsPartial", [])
        age = None
        gender = None
        for trial in trials:
            # Check direct age
            if "age" in trial and str(trial["age"]).isdigit():
                age = int(trial["age"])
            # Check age/gender inside response dict
            elif isinstance(trial.get("response"), dict):
                resp = trial["response"]
                if "Q0" in resp:
                    q0_val = resp["Q0"].strip().lower()
                    # If numeric, it's probably age
                    if q0_val.isdigit():
                        age = int(q0_val)
                    # If "male"/"female", it's gender
                    elif q0_val in ["male", "female"]:
                        gender = q0_val
            # Direct gender field
            if "gender" in trial and isinstance(trial["gender"], str):
                g = trial["gender"].strip().lower()
                if g in ["male", "female"]:
                    gender = g
        # Only include if both age & gender are present
        if age is not None and gender is not None:
            ages.append(age)
            if gender == "female":
                total_females += 1
            elif gender == "male":
                total_males += 1
        else:
            excluded.append(pid)
    included_count = len(ages)
    total_other = included_count - total_females - total_males
    # Stats
    if ages:
        median_age = np.median(ages)
        mean_age = np.mean(ages)
        std_age = np.std(ages, ddof=1)
        min_age = np.min(ages)
        max_age = np.max(ages)
    else:
        median_age = mean_age = std_age = min_age = max_age = None
    return {
        "total_included": included_count,
        "total_females": total_females,
        "total_males": total_males,
        "total_other": total_other,
        "median_age": median_age,
        "mean_age": mean_age,
        "std_age": std_age,
        "min_age": min_age,
        "max_age": max_age
    }

def average_judgments_per_word_pair_exp2(data):
    """
    Calculate the average number of judgments per word pair in Experiment 2.
    Returns the average and detailed statistics.
    """
    # Dictionary to count judgments per word pair
    word_pair_counts = defaultdict(int)
    total_judgments = 0
    
    participants = data["__collections__"]["exptDataHumanClue"]
    
    for pid, pdata in participants.items():
        trials = pdata.get("trialsPartial", [])
        
        for trial in trials:
            # Check if this trial has word pair judgment data
            if ("word1" in trial and "word2" in trial and 
                "response" in trial and 
                trial.get("exp_phase") == "trial"):
                
                word1 = trial["word1"].strip().lower()
                word2 = trial["word2"].strip().lower()
                
                # Create consistent word pair key (alphabetical order)
                word_pair = tuple(sorted([word1, word2]))
                
                word_pair_counts[word_pair] += 1
                total_judgments += 1
    
    # Calculate statistics
    if word_pair_counts:
        judgment_counts = list(word_pair_counts.values())
        avg_judgments = np.mean(judgment_counts)
        median_judgments = np.median(judgment_counts)
        std_judgments = np.std(judgment_counts, ddof=1)
        min_judgments = np.min(judgment_counts)
        max_judgments = np.max(judgment_counts)
        total_unique_pairs = len(word_pair_counts)
    else:
        avg_judgments = median_judgments = std_judgments = 0
        min_judgments = max_judgments = total_unique_pairs = 0
    
    return {
        "total_unique_word_pairs": total_unique_pairs,
        "total_judgments": total_judgments,
        "average_judgments_per_pair": avg_judgments,
        "median_judgments_per_pair": median_judgments,
        "std_judgments_per_pair": std_judgments,
        "min_judgments_per_pair": min_judgments,
        "max_judgments_per_pair": max_judgments,
        "word_pair_counts": dict(word_pair_counts)  # Optional: detailed breakdown
    }

def average_judgments_per_participant_exp2(data):
    """
    Calculate the average number of pairwise judgments each participant made in Experiment 2.
    Returns statistics about judgments per participant.
    """
    participant_judgment_counts = []
    participants = data["__collections__"]["exptDataHumanClue"]
    
    for pid, pdata in participants.items():
        trials = pdata.get("trialsPartial", [])
        judgment_count = 0
        
        for trial in trials:
            # Check if this trial has word pair judgment data
            if ("word1" in trial and "word2" in trial and 
                "response" in trial and 
                trial.get("exp_phase") == "trial"):
                judgment_count += 1
        
        if judgment_count > 0:  # Only include participants who made judgments
            participant_judgment_counts.append(judgment_count)
    
    # Calculate statistics
    if participant_judgment_counts:
        avg_judgments = np.mean(participant_judgment_counts)
        median_judgments = np.median(participant_judgment_counts)
        std_judgments = np.std(participant_judgment_counts, ddof=1)
        min_judgments = np.min(participant_judgment_counts)
        max_judgments = np.max(participant_judgment_counts)
        total_participants = len(participant_judgment_counts)
        total_judgments = sum(participant_judgment_counts)
    else:
        avg_judgments = median_judgments = std_judgments = 0
        min_judgments = max_judgments = total_participants = total_judgments = 0
    
    return {
        "total_participants_with_judgments": total_participants,
        "total_judgments_made": total_judgments,
        "average_judgments_per_participant": avg_judgments,
        "median_judgments_per_participant": median_judgments,
        "std_judgments_per_participant": std_judgments,
        "min_judgments_per_participant": min_judgments,
        "max_judgments_per_participant": max_judgments,
        "participant_judgment_counts": participant_judgment_counts  # Optional: detailed breakdown
    }

# Your existing code
with open('experiment-1/experiment-1-subjects.json') as f:
    exp1 = json.load(f)
with open('experiment-2/exptDataHumanClue.json') as f:
    exp2 = json.load(f)

print("Experiment 1:", count_exp_1(exp1))
print("Experiment 2:", count_exp_2(exp2))

# Word pair analysis
word_pair_stats = average_judgments_per_word_pair_exp2(exp2)
print("\n--- Word Pair Analysis ---")
print("Average judgments per word pair:", word_pair_stats["average_judgments_per_pair"])
print("Total unique word pairs:", word_pair_stats["total_unique_word_pairs"])
print("Total judgments:", word_pair_stats["total_judgments"])
print("Min/Max judgments per pair:", word_pair_stats["min_judgments_per_pair"], "/", word_pair_stats["max_judgments_per_pair"])

# Participant analysis
participant_stats = average_judgments_per_participant_exp2(exp2)
print("\n--- Participant Analysis ---")
print("Average judgments per participant:", participant_stats["average_judgments_per_participant"])
print("Total participants with judgments:", participant_stats["total_participants_with_judgments"])
print("Median judgments per participant:", participant_stats["median_judgments_per_participant"])
print("Min/Max judgments per participant:", participant_stats["min_judgments_per_participant"], "/", participant_stats["max_judgments_per_participant"])