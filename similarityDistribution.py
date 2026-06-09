import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Create static folder if it doesn't exist
os.makedirs("static", exist_ok=True)

# Load recommender data
with open("music_recommender.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
X_scaled = data["X_scaled"]

# Number of songs to sample
sample_size = min(500, len(X_scaled))

scores = []

print("Calculating similarity scores...")

for idx in range(sample_size):

    distances, indices = model.kneighbors(
        X_scaled[idx].reshape(1, -1),
        n_neighbors=11
    )

    # Ignore the first result (the song itself)
    similarities = 1 - distances[0][1:]

    scores.extend(similarities)

scores = np.array(scores)

# Statistics
avg_similarity = np.mean(scores)
min_similarity = np.min(scores)
max_similarity = np.max(scores)

print(f"Average Similarity: {avg_similarity:.4f}")
print(f"Minimum Similarity: {min_similarity:.4f}")
print(f"Maximum Similarity: {max_similarity:.4f}")

# Create histogram
plt.figure(figsize=(10, 6))

plt.hist(
    scores,
    bins=30
)

plt.title("Similarity Score Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")

# Add average line
plt.axvline(
    avg_similarity,
    linestyle="--",
    linewidth=2,
    label=f"Average = {avg_similarity:.2f}"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "static/similarity_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print("static/similarity_distribution.png")