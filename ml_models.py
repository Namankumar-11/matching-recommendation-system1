import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Load data
users = pd.read_csv("../data/users.csv")
interactions = pd.read_csv("../data/interactions.csv")

# Label
positive_actions = ["swipe_like", "message", "reply"]
interactions["label"] = interactions["action"].isin(positive_actions).astype(int)

# Merge source + target user features
data = interactions.merge(users, left_on="user_id", right_on="user_id", suffixes=("","_u"))
data = data.merge(users, left_on="target_user", right_on="user_id", suffixes=("","_t"))

# Feature Engineering
data["age_diff"] = abs(data["age"] - data["age_t"])
data["lifestyle_match"] = (data["lifestyle"] == data["lifestyle_t"]).astype(int)
data["intent_match"] = (data["intent"] == data["intent_t"]).astype(int)
data["education_match"] = (data["education"] == data["education_t"]).astype(int)

data["compatibility_score"] = (
    3 * data["intent_match"] +
    2 * data["lifestyle_match"] +
    2 * (data["age_diff"] <= 3).astype(int) +
    1 * data["education_match"]
)

# Historical behavior features
user_like_rate = interactions.groupby("user_id")["label"].mean()
target_like_rate = interactions.groupby("target_user")["label"].mean()

data["user_like_rate"] = data["user_id"].map(user_like_rate).fillna(0)
data["target_like_rate"] = data["target_user"].map(target_like_rate).fillna(0)

# Feature set
features = [
    "age_diff", "lifestyle_match", "intent_match",
    "education_match", "compatibility_score",
    "user_like_rate", "target_like_rate"
]

X = data[features]
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, preds)

print(f"🔥 FINAL MODEL ROC-AUC: {auc:.4f}")

joblib.dump(model, "match_predictor.pkl")
print("✅ Production Model Saved")