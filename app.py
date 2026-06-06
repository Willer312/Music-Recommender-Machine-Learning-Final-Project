import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)



@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

df["display_name"] = (
    df["name"].astype(str)
    + " - "
    + df["artists"].astype(str)
)



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

@st.cache_resource
def build_model():

    X = df[FEATURES]

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



st.sidebar.title("📊 Dataset Information")

st.sidebar.metric(
    "Songs",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Features Used",
    len(FEATURES)
)

st.sidebar.markdown("---")

st.sidebar.write("""
### Machine Learning Model

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


st.title("🎵 Music Recommendation System")

st.write("""
This application recommends songs based on Spotify
audio features using K-Nearest Neighbors (KNN)
and Cosine Similarity.
""")



tab1, tab2, tab3 = st.tabs([
    "🎵 Recommender",
    "📊 Dataset & Methodology",
    "👥 About Team"
])



with tab1:

    st.subheader("Find Similar Songs")

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

            st.warning(
                "No matching songs found."
            )

    if st.button(
        "🎧 Recommend Songs",
        use_container_width=True
    ):

        if not selected_song:

            st.warning(
                "Please search and select a song first."
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

                similarity = (1 - distance) * 100

                st.markdown(
                    f"### #{rank} {df.iloc[song_idx]['name']}"
                )

                st.caption(
                    f"{df.iloc[song_idx]['artists']}"
                )

                st.progress(
                    float(similarity / 100)
                )

                st.write(
                    f"Similarity: {similarity:.2f}%"
                )

                st.divider()



with tab2:

    st.header("Dataset Overview")

    st.write(f"""
    This project uses a Spotify dataset containing
    **{len(df):,} songs** and 19 attributes.
    """)

    st.subheader("Features Used")

    st.markdown("""
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

    st.subheader("Data Preprocessing")

    st.write("""
    The selected audio features are standardized
    using StandardScaler before training.

    Standardization ensures that features with
    larger numerical ranges do not dominate
    similarity calculations.
    """)

    st.subheader("Machine Learning Method")

    st.write("""
    The recommendation engine uses the
    K-Nearest Neighbors (KNN) algorithm.

    Each song is represented as a numerical vector
    using Spotify audio features.

    When a song is selected, cosine similarity
    is used to identify the most similar songs
    in the dataset.
    """)

    st.subheader("Cosine Similarity")

    st.latex(
        r"\frac{A \cdot B}{||A|| ||B||}"
    )

    st.write("""
    Songs with similar audio profiles produce
    higher cosine similarity scores and are
    recommended to the user.
    """)

    st.subheader("Feature Statistics")

    st.dataframe(
        df[FEATURES].describe()
    )

    st.subheader("Dataset Sample")

    st.dataframe(
        df.head(10)
    )


with tab3:

    st.header("Project Team")

    st.markdown("""
    ### Member 1
    **Name:** Willian Yehezkiel Alvin

    **Responsibilities**
    - Data preprocessing
    - Feature engineering
    - KNN implementation

    ---

    ### Member 2
    **Name:** Vittorio Dinata

    **Responsibilities**
    - Streamlit development
    - User interface design
    - Application testing

    ---

    ### Member 3
    **Name:** Gregorius Gilbert Susanto

    **Responsibilities**
    - Documentation
    - Literature review
    - Report writing

    ---

    ### Member 4
    **Name:** Andrew Ong

    **Responsibilities**
    - Dataset analysis
    - Evaluation and validation
    - Presentation preparation
    """)

    st.subheader("Project Objective")

    st.write("""
    The purpose of this project is to develop a
    content-based music recommendation system using
    Spotify audio features.

    The recommendation engine utilizes the
    K-Nearest Neighbors (KNN) algorithm and
    cosine similarity to identify songs with
    similar musical characteristics.

    By analyzing audio features such as
    danceability, energy, valence, acousticness,
    and tempo, the system generates personalized
    song recommendations for users.
    """)

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    - K-Nearest Neighbors (KNN)
    - Cosine Similarity
    """)
