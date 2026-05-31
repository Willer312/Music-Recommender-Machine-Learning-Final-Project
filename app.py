import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# ====================================
# CUSTOM CSS
# ====================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.song-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    border-left: 5px solid #1DB954;
}

.song-title {
    font-size: 22px;
    font-weight: bold;
}

.artist {
    color: #b3b3b3;
}

.similarity {
    color: #1DB954;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# LOAD DATA
# ====================================

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# Create display names

df["display_name"] = (
    df["name"].astype(str)
    + " - "
    + df["artists"].astype(str)
)

# ====================================
# BUILD MODEL
# ====================================

@st.cache_resource
def build_model():

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

    model = NearestNeighbors(
        n_neighbors=11,
        metric='cosine',
        algorithm='brute'
    )

    model.fit(X_scaled)

    return model, X_scaled

model, X_scaled = build_model()

# ====================================
# SIDEBAR
# ====================================

st.sidebar.title("📊 Dataset Information")

st.sidebar.metric(
    "Songs",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Features",
    "9"
)

st.sidebar.markdown("---")

st.sidebar.write("""
### Model

- K-Nearest Neighbors (KNN)
- Cosine Similarity
- Content-Based Filtering

### Audio Features

- Danceability
- Energy
- Acousticness
- Instrumentalness
- Liveness
- Speechiness
- Valence
- Loudness
- Tempo
""")

# ====================================
# TITLE
# ====================================

st.title("🎵 Music Recommendation System")

st.write(
    "Find similar songs using Spotify audio features, "
    "K-Nearest Neighbors (KNN), and Cosine Similarity."
)

st.markdown("---")

# ====================================
# SONG SEARCH
# ====================================

search_query = st.text_input(
    "🔍 Search Song",
    placeholder="Type part of a song title..."
)

selected_song = None

if search_query:

    matches = df[
        df["name"].str.contains(
            search_query,
            case=False,
            na=False
        )
    ]

    matches = matches.head(20)

    if len(matches) > 0:

        selected_song = st.selectbox(
            "Select a Song",
            matches["display_name"]
        )

    else:

        st.warning("No matching songs found.")

# ====================================
# RECOMMEND BUTTON
# ====================================

if st.button(
    "🎧 Recommend Songs",
    use_container_width=True
):

    if not selected_song:

        st.warning(
            "Please search for and select a song first."
        )

    else:

        selected_row = df[
            df["display_name"] == selected_song
        ].iloc[0]

        idx = selected_row.name

        st.success(
            f"Selected Song: "
            f"{selected_row['name']} - "
            f"{selected_row['artists']}"
        )

        distances, indices = model.kneighbors(
            X_scaled[idx].reshape(1, -1),
            n_neighbors=11
        )

        st.subheader("🎶 Recommended Songs")

        for rank, (distance, song_idx) in enumerate(
            zip(
                distances[0][1:],
                indices[0][1:]
            ),
            start=1
        ):

            similarity = (
                (1 - distance) * 100
            )

            st.markdown(
                f"""
                <div class="song-card">

                    <div class="song-title">
                        #{rank} {df.iloc[song_idx]['name']}
                    </div>

                    <div class="artist">
                        {df.iloc[song_idx]['artists']}
                    </div>

                    <div class="similarity">
                        Similarity:
                        {similarity:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(similarity / 100)
            )

