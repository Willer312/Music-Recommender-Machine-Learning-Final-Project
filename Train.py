import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("data.csv")

print(f"Dataset loaded: {df.shape[0]} songs")


features = [
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

X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features standardized.")



model = NearestNeighbors(
    n_neighbors=11,
    metric='cosine',
    algorithm='brute'
)

model.fit(X_scaled)

print("KNN model trained.")



def recommend(song_name, artist_name, top_n=10):

    matches = df[
        (df['name'].str.lower() == song_name.lower()) &
        (df['artists'].str.lower().str.contains(artist_name.lower()))
    ]

    if len(matches) == 0:
        print("Song not found.")
        return

    idx = matches.index[0]

    distances, indices = model.kneighbors(
        X_scaled[idx].reshape(1, -1),
        n_neighbors=top_n + 1
    )

    print(f"\nRecommendations for:")
    print(f"{df.iloc[idx]['name']} - {df.iloc[idx]['artists']}\n")

    for rank, (distance, song_idx) in enumerate(
        zip(distances[0][1:], indices[0][1:]),
        start=1
    ):

        similarity = 1 - distance

        print(
            f"{rank}. "
            f"{df.iloc[song_idx]['name']} - "
            f"{df.iloc[song_idx]['artists']} "
            f"(Similarity: {similarity:.4f})"
        )


while True:

    song_name = input("\nSong title (or 'quit'): ")

    if song_name.lower() == "quit":
        break

    artist_name = input("Artist: ")

    recommend(song_name, artist_name)
