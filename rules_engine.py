import pandas as pd
import numpy as np

# ---------------- LOAD DATA ----------------
users = pd.read_csv("../data/users.csv")
prefs = pd.read_csv("../data/preferences.csv")

# ---------------- HELPER FUNCTIONS ----------------

def age_match(user_age, pref_min, pref_max):
    return 1 if pref_min <= user_age <= pref_max else 0

def height_match(user_height, pref_height):
    diff = abs(user_height - pref_height)
    return max(0, 1 - diff / 30)

def lifestyle_match(l1, l2):
    return 1 if l1 == l2 else 0.5

def intent_match(i1, i2):
    return 1 if i1 == i2 else 0

# ---------------- SCORING FUNCTION ----------------

def compute_match_score(u1, u2, p1):
    score = 0
    
    score += 0.30 * age_match(u2["age"], p1["pref_age_min"], p1["pref_age_max"])
    score += 0.25 * height_match(u2["height"], p1["pref_height"])
    score += 0.20 * lifestyle_match(u1["lifestyle"], u2["lifestyle"])
    score += 0.25 * intent_match(u1["intent"], u2["intent"])
    
    return round(score, 3)

# ---------------- RECOMMENDATION ENGINE ----------------

def recommend_matches(user_id, top_k=10):
    user = users[users["user_id"] == user_id].iloc[0]
    pref = prefs[prefs["user_id"] == user_id].iloc[0]
    
    candidates = users[users["gender"] != user["gender"]]
    
    scores = []
    
    for _, target in candidates.iterrows():
        score = compute_match_score(user, target, pref)
        scores.append((target["user_id"], score))
    
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return ranked[:top_k]

# ---------------- TEST ----------------

if __name__ == "__main__":
    print("Top matches for user 10:\n")
    print(recommend_matches(10, top_k=10))