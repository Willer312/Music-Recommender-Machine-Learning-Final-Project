import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("static", exist_ok=True)

with open("music_recommender.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]

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

plt.figure(figsize=(10,8))

sns.heatmap(
    df[FEATURES].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "static/heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()