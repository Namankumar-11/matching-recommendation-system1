import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
interactions = pd.read_csv("../data/interactions.csv")

# Convert actions to weights
action_weights = {
    "view": 1,
    "swipe_like": 3,
    "message": 5,
    "reply": 6
}

interactions["weight"] = interactions["action"].map(action_weights)

# Build User-User interaction matrix
matrix = interactions.pivot_table(
    index="user_id",
    columns="target_user",
    values="weight",
    aggfunc="sum",
    fill_value=0
)

# Compute similarity between users
user_similarity = cosine_similarity(matrix)

# ---------------- Recommendation Engine ----------------

def recommend_by_behavior(user_id, top_k=10):
    user_index = matrix.index.get_loc(user_id)
    
    similarity_scores = list(enumerate(user_similarity[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    similar_users = [matrix.index[i] for i, score in similarity_scores[1:6]]
    
    candidate_scores = matrix.loc[similar_users].sum().sort_values(ascending=False)
    
    return candidate_scores.head(top_k)

# ---------------- Test ----------------

if __name__ == "__main__":
    print("Behaviour-based recommendations for user 10:\n")
    print(recommend_by_behavior(10))