import random
import pandas as pd
from faker import Faker

fake = Faker()

# -------------------- USER DATA GENERATION --------------------

N_USERS = 2000

users = []

lifestyles = ["active", "moderate", "sedentary"]
intents = ["friendship", "dating", "serious"]
educations = ["school", "bachelor", "master", "phd"]

for i in range(N_USERS):
    users.append({
        "user_id": i,
        "age": random.randint(18, 35),
        "gender": random.choice(["M", "F"]),
        "lifestyle": random.choice(lifestyles),
        "intent": random.choice(intents),
        "education": random.choice(educations)
    })

users_df = pd.DataFrame(users)

users_df.to_csv("users.csv", index=False)

print("✅ users.csv generated")

N_INTERACTIONS = 50000
N_USERS = len(users_df)

interactions = []

for _ in range(N_INTERACTIONS):

    u1 = random.randint(0, N_USERS - 1)
    u2 = random.randint(0, N_USERS - 1)

    if u1 == u2:
        continue

    user1 = users_df.iloc[u1]
    user2 = users_df.iloc[u2]

    compatibility = 0

    # Age compatibility
    if abs(user1["age"] - user2["age"]) <= 3:
        compatibility += 2

    # Lifestyle match
    if user1["lifestyle"] == user2["lifestyle"]:
        compatibility += 2

    # Relationship intent match
    if user1["intent"] == user2["intent"]:
        compatibility += 3

    # Education match
    if user1["education"] == user2["education"]:
        compatibility += 1

    # Action generation based on compatibility
    if compatibility >= 6:
        action = random.choices(
            ["swipe_like", "message", "reply"],
            weights=[0.35, 0.40, 0.25]
        )[0]

    elif compatibility >= 3:
        action = random.choices(
            ["view", "swipe_like", "swipe_pass"],
            weights=[0.40, 0.35, 0.25]
        )[0]

    else:
        action = random.choices(
            ["view", "swipe_pass"],
            weights=[0.25, 0.75]
        )[0]

    interactions.append({
        "user_id": u1,
        "target_user": u2,
        "action": action,
        "timestamp": fake.date_time_this_year()
    })

interactions_df = pd.DataFrame(interactions)

interactions_df.to_csv("interactions.csv", index=False)