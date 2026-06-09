import pickle
import pandas as pd

FEATURES = [
    'danceability',
    'energy',
    'acousticness',
    'instrumentalness',
    'liveness',
    'speechiness',
    'valence',
    'loudness',
    'tempo'
]

with open("music_recommender.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]

import matplotlib.pyplot as plt

df[FEATURES].hist(
    figsize=(14,10),
    bins=30
)

plt.tight_layout()
plt.savefig("static/eda.png", dpi=300)
plt.show()