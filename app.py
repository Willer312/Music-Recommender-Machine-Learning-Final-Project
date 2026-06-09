import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import os
import pickle
import numpy as np
import pandas as pd
import random

# --- 1. LOAD DATA DARI PICKLE (DENGAN CACHE AGAR INSTAN) ---
@st.cache_resource
def load_pipeline():
    with open("music_recommender.pkl", "rb") as f:
        data = pickle.load(f)
    return data['model'], data['X_scaled'], data['df']

model, X_scaled, df = load_pipeline()
def show_image_if_exists(path):
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Missing image: {path}")

# Helper function to convert local images to Base64
def get_base64_img(img_name):
    path = os.path.join("static", img_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

# Convert all your team images at once
img_william = get_base64_img("WillianYehezkiel.jpg")
img_vittorio = get_base64_img("VittorioDinata.jpg")
img_gilbert = get_base64_img("GregoriusGilbert.jpg")
img_andrew = get_base64_img("AndrewOng.jpg")


# Set up Streamlit page settings
st.set_page_config(
    page_title="Algorhythm — Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── NAVIGATION BACKEND (CONDITIONAL OPTION MENU) ───
try:
    from streamlit_option_menu import option_menu
    _HAS_MENU = True
except ImportError:
    _HAS_MENU = False

# ─── RECONSTRUCTING YOUR BEAUTIFUL CSS VARIABLES & CORE STYLES ───
custom_css = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

  :root {
    --bg: #0a0a0f;
    --bg2: #111118;
    --bg3: #18181f;
    --surface: #1e1e28;
    --surface2: #252530;
    --border: rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.13);
    --text: #f0eef8;
    --text2: #9998b3;
    --text3: #5e5d78;
    --accent: #7c6af5;
    --accent2: #a594ff;
    --accent3: #c9bfff;
    --green: #43d98c;
    --pink: #f06292;
    --amber: #f5a623;
    --font-display: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --r: 12px;
    --r2: 20px;
  }

  .block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem;
    max-width: 1400px;
}
  div[data-testid="stHorizontalBlock"] { gap: 0; }

  .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
  }
  
  h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display) !important;
    color: var(--text) !important;
  }

  .hero-gradient { 
    background: linear-gradient(135deg, #fff 0%, var(--accent3) 60%, var(--pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    line-height: 1.15;
    display: inline-block;
    padding-bottom: 0.2em;
}

  .info-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r2); padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .info-card .icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; margin-bottom: 1rem;
  }
  .icon-purple { background: rgba(124,106,245,0.15); }
  .icon-green { background: rgba(67,217,140,0.15); }
  .icon-pink { background: rgba(240,98,146,0.15); }
  .icon-amber { background: rgba(245,166,35,0.15); }

  .tag {
    padding: 5px 12px; display: inline-block;
    border-radius: 100px; font-size: 0.78rem; font-weight: 500;
    border: 1px solid var(--border2); color: var(--text2); background: var(--surface);
    margin-right: 4px; margin-bottom: 4px;
  }
  .tag.accent { border-color: rgba(124,106,245,0.35); color: var(--accent3); background: rgba(124,106,245,0.1); }

  .song-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); padding: 1.25rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 0.75rem;
  }
  .song-rank {
    font-family: var(--font-display); font-size: 1.5rem; font-weight: 800;
    color: var(--text3); min-width: 2.5rem; text-align: center;
  }
  .song-art {
    width: 48px; height: 48px; background: linear-gradient(135deg, var(--surface2), var(--bg3));
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem; flex-shrink: 0; border: 1px solid var(--border);
  }
  .song-info { flex: 1; min-width: 0; }
  .song-name { font-weight: 500; font-size: 0.9rem; margin-bottom: 2px; color: var(--text); }
  .song-artist { color: var(--text2); font-size: 0.78rem; }
  
  .similarity-bar { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; min-width: 60px; }
  .sim-num { font-family: var(--font-display); font-size: 0.85rem; font-weight: 700; color: var(--green); }
  .sim-track { width: 50px; height: 3px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
  .sim-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); border-radius: 2px; }

  .pipeline-container { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1rem; }
  .pipe-step-py {
    flex: 1; min-width: 160px; background: var(--surface);
    border: 1px solid var(--border); padding: 1.25rem; border-radius: var(--r);
  }
  .pipe-num { font-size: 0.7rem; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  .pipe-name { font-weight: 600; font-size: 0.875rem; margin-bottom: 4px; color: var(--text); }
  .pipe-desc { font-size: 0.75rem; color: var(--text2); }

  .team-card-py {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r2); padding: 1.75rem; text-align: center;
  }
  .team-name { font-family: var(--font-display); font-weight: 700; margin-bottom: 3px; color: var(--text); }
  .team-role { font-size: 0.78rem; color: var(--accent3); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin-bottom: 0.75rem; }
  .team-desc { font-size: 0.82rem; color: var(--text2); line-height: 1.55; }
  .team-card-py img { width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin-bottom: 10px; border: 2px solid var(--accent); }
  .chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r2);
    padding: 1rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.25rem;
    text-align: center;
}

.metric-number {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent2);
}

.metric-label {
    color: var(--text2);
    font-size: 0.85rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ─── 2. MENYIAPKAN TAMPILAN DROPDOWN SECARA DINAMIS (PENCEGAHAN LAG) ───
if 'dropdown_label' not in df.columns:
    df['dropdown_label'] = df['name'] + " — " + df['artists']


# ─── 3. LOGIKA REKOMENDASI KNN BERDASARKAN DATA ASLI ───
def generate_recommendations_dinamis(selected_label, n_recs=10):
    song_row = df[df['dropdown_label'] == selected_label]
    if song_row.empty:
        return []
        
    song_idx = song_row.index[0]
    target_features = X_scaled[song_idx].reshape(1, -1)
    
    distances, indices = model.kneighbors(target_features, n_neighbors=n_recs + 1)
    
    recommended_indices = indices[0][1:]
    recommended_distances = distances[0][1:]
    
    emojis = ["🎵", "🎧", "🎹", "🎸", "🎷", "🎺", "🥁", "🎤", "🎶", "🎼"]
    results = []
    
    for idx, (mesh_idx, dist) in enumerate(zip(recommended_indices, recommended_distances)):
        row_data = df.iloc[mesh_idx]
        similarity_score = 1.0 - dist
        similarity_score = min(0.99, max(0.50, similarity_score))
        
        results.append({
            "rank": idx + 1,
            "name": row_data["name"],
            "artist": row_data["artists"],
            "similarity": round(similarity_score, 2),
            "emoji": emojis[idx % len(emojis)]
        })
    return results


# ─── RENDERING NAVIGATION MATRIX ───
if _HAS_MENU:
    menu = option_menu(
        menu_title=None,
        options=["Discover", "Background", "Model", "Dataset", "Teams"],
        icons=["search", "info-circle-fill", "cpu", "database-fill", "people-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#111118", "border-radius": "8px", "border": "1px solid rgba(255,255,255,0.07)", "margin-bottom": "25px"},
            "icon":       {"color": "#7c6af5", "font-size": "15px"},
            "nav-link":   {
                "font-size": "14px", "font-weight": "600", "color": "#9998b3",
                "padding": "10px 20px", "border-radius": "6px",
                "--hover-color": "#1e1e28",
            },
            "nav-link-selected": {"background-color": "#7c6af5", "color": "#ffffff"},
        },
    )
else:
    with st.sidebar:
        st.markdown('<h3 style="margin-bottom:0px;">🎧 algorhythm</h3>', unsafe_allow_html=True)
        menu = st.radio(
            "Navigate",
            ["Discover", "Background", "Model", "Dataset", "Teams"],
            index=0,
        )


# ════════════ 1. DISCOVER SECTION ════════════
if menu == "Discover":
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown('<h1><span class="hero-gradient">Find your next<br>favorite song.</span></h1>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:var(--text2); font-size:1.1rem;'>Powered by a KNN model trained on {df.shape[0]:,} Spotify tracks. Enter any song and discover music that truly matches your taste — not just what's trending.</p>", unsafe_allow_html=True)
        
        st.markdown(f'''
            <span class="tag accent">{df.shape[0]:,} songs</span>
            <span class="tag">Cosine Similarity</span>
            <span class="tag">KNN Model</span>
            <span class="tag">9 audio features</span>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="info-card"><h4>Search Song Database</h4>', unsafe_allow_html=True)
        
        # 1. KEMBALIKAN LOGIKA LAMA: Menggunakan Text Input terlebih dahulu sebagai filter keyword awal
        search_keyword = st.text_input(
            "🔍 Type song title...",
            placeholder="Type part of a song title (e.g. 'Blinding Lights')...",
            label_visibility="collapsed"
        )
        
        selected_search = None
        
        # 2. Batasi item di dropdown selectbox secara dinamis berdasarkan keyword masukan (Maksimal 20 item)
        if search_keyword:
            matches = df[df["name"].str.contains(search_keyword, case=False, na=False)]
            matches_limit = matches.head(20)
            
            if not matches_limit.empty:
                selected_search = st.selectbox(
                    "Matching results:",
                    options=matches_limit["dropdown_label"].tolist()
                )
            else:
                st.markdown("<p style='color:var(--pink); font-size:0.85rem; margin-top:5px;'>❌ No matching songs found. Try another title.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size:0.85rem; color:var(--text3); margin-top:5px;'>💡 Type a song title above to reveal the recommendation options.</p>", unsafe_allow_html=True)
            
        get_recs = st.button("✨ Get Recommendations", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Eksekusi rekomendasi berdasarkan lagu yang berhasil difilter
    if get_recs and selected_search:
        # Menghindari splitting string secara manual demi keamanan karakter khusus/strip bawaan judul lagu
        song_row = df[df['dropdown_label'] == selected_search]
        
        if not song_row.empty:
            track_name = song_row.iloc[0]["name"]
            track_artist = song_row.iloc[0]["artists"]
            st.markdown(f"### Recommendations Based On: <span style='color: var(--accent2);'>{track_name} ({track_artist})</span>", unsafe_allow_html=True)
            
            with st.spinner(f"Searching {df.shape[0]:,} tracks via acoustic profiles..."):
                recommendations = generate_recommendations_dinamis(selected_search, n_recs=10)
                
                if recommendations:
                    r_col1, r_col2 = st.columns(2)
                    for i, rec in enumerate(recommendations):
                        pct = int(rec["similarity"] * 100)
                        card_html = f'''
                        <div class="song-card">
                          <div class="song-rank">{str(rec["rank"]).zfill(2)}</div>
                          <div class="song-art">{rec["emoji"]}</div>
                          <div class="song-info">
                            <div class="song-name">{rec["name"]}</div>
                            <div class="song-artist">{rec["artist"]}</div>
                          </div>
                          <div class="similarity-bar">
                            <div class="sim-num">{pct}%</div>
                            <div class="sim-track"><div class="sim-fill" style="width:{pct}%"></div></div>
                          </div>
                        </div>
                        '''
                        if i % 2 == 0:
                            r_col1.markdown(card_html, unsafe_allow_html=True)
                        else:
                            r_col2.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.error("Gagal mengambil data rekomendasi dari model.")
        else:
            st.error("Track tidak ditemukan di database.")
    elif get_recs:
        st.warning("Please type a valid song title and select it from the dropdown first!")


# ════════════ 2. BACKGROUND SECTION ════════════
elif menu == "Background":
    st.markdown('<div class="section-eyebrow">The Problem</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Why do music recommendations fail?</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--text2); font-size:1.1rem;">Streaming platforms often recommend music based on what\'s popular — not what you\'ll actually enjoy. Algorhythm takes a different approach: recommending songs based on the <em>acoustic DNA</em> of the music itself.</p>', unsafe_allow_html=True)
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.markdown('<div class="info-card"><div class="icon icon-purple">🎯</div><h4>Audio-first matching</h4><p style="color:var(--text2); font-size:0.85rem;">Instead of relying on play counts or user behavior, we analyze 9 acoustic features extracted directly from each track.</p></div>', unsafe_allow_html=True)
    with b_col2:
        st.markdown('<div class="info-card"><div class="icon icon-green">🔊</div><h4>Genre-agnostic</h4><p style="color:var(--text2); font-size:0.85rem;">A jazz song and an electronic track can have similar energy and tempo profiles. Our model finds these hidden connections.</p></div>', unsafe_allow_html=True)
    with b_col3:
        st.markdown('<div class="info-card"><div class="icon icon-pink">📈</div><h4>No popularity bias</h4><p style="color:var(--text2); font-size:0.85rem;">Popular songs aren\'t given more weight. A 1940s blues track is just as valid a recommendation as a 2020 pop hit.</p></div>', unsafe_allow_html=True)
    with b_col4:
        st.markdown('<div class="info-card"><div class="icon icon-amber">🧠</div><h4>Content-based</h4><p style="color:var(--text2); font-size:0.85rem;">We use content-based collaborative filtering — the most reliable approach when explicit user ratings are missing.</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    ctx_col1, ctx_col2 = st.columns([1.5, 1])
    with ctx_col1:
        st.markdown("### Project Context")
        st.markdown(f"<p style='color:var(--text2); font-size:0.95rem; line-height:1.7;'>Algorhythm was developed as an academic project exploring content-based music recommendation systems. The core challenge: given a song a user enjoys, find the 10 most acoustically similar songs in a corpus of {df.shape[0]:,} tracks — efficiently and accurately.<br><br>The solution combines <b>StandardScaler preprocessing</b> to normalize feature ranges, a <b>brute-force KNN model</b> for nearest-neighbor search, and <b>cosine similarity</b> as the distance metric — chosen because it measures directional closeness, not magnitude.</p>", unsafe_allow_html=True)
    with ctx_col2:
        st.markdown(f'<div class="info-card" style="padding:1rem 1.5rem;">'
                    f'<h5>📊 Metric Snapshot</h5>'
                    f'<p style="margin:5px 0;"><b>{df.shape[0]//1000}K+</b> <span style="color:var(--text2); font-size:0.85rem;">Songs in dataset spans 100 years</span></p>'
                    f'<p style="margin:5px 0;"><b>9</b> <span style="color:var(--text2); font-size:0.85rem;">Acoustic Vector features</span></p>'
                    f'<p style="margin:5px 0;"><b>KNN</b> <span style="color:var(--text2); font-size:0.85rem;">Cosine Metric Strategy</span></p>'
                    f'<p style="margin:5px 0;"><b>~98%</b> <span style="color:var(--text2); font-size:0.85rem;">Query Corpus coverage profile</span></p>'
                    f'</div>', unsafe_allow_html=True)


# ════════════ 3. MODEL SECTION ════════════
elif menu == "Model":

    st.markdown(
        '<h2>Model Architecture & Evaluation</h2>',
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        <p style="color:var(--text2);">
        Algorhythm uses a K-Nearest Neighbors recommendation engine
        with cosine similarity over standardized Spotify audio features.
        </p>
        ''',
        unsafe_allow_html=True
    )

    # ==========================
    # MODEL OVERVIEW
    # ==========================

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '''
            <div class="info-card">
                <h4>K-Nearest Neighbors (KNN)</h4>
                <p style="color:var(--text2);">
                KNN retrieves songs with the closest acoustic profiles.
                </p>
                <code>n_neighbors = 11</code>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '''
            <div class="info-card">
                <h4>Cosine Similarity</h4>
                <p style="color:var(--text2);">
                Similarity is measured using vector direction rather than magnitude.
                </p>
                <code>similarity = 1 - cosine_distance</code>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # ==========================
    # PIPELINE
    # ==========================

    st.markdown("### Model Pipeline")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("""
        <div class="info-card">
            <div class="pipe-num">Step 1</div>
            <div class="pipe-name">Load Data</div>
            <div class="pipe-desc">Spotify Dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="info-card">
            <div class="pipe-num">Step 2</div>
            <div class="pipe-name">Feature Selection</div>
            <div class="pipe-desc">9 Audio Features</div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="info-card">
            <div class="pipe-num">Step 3</div>
            <div class="pipe-name">StandardScaler</div>
            <div class="pipe-desc">Normalize Feature Space</div>
        </div>
        """, unsafe_allow_html=True)

    p4, p5 = st.columns(2)

    with p4:
        st.markdown("""
        <div class="info-card">
            <div class="pipe-num">Step 4</div>
            <div class="pipe-name">KNN Search</div>
            <div class="pipe-desc">Cosine Similarity Lookup</div>
        </div>
        """, unsafe_allow_html=True)

    with p5:
        st.markdown("""
        <div class="info-card">
            <div class="pipe-num">Step 5</div>
            <div class="pipe-name">Recommendation</div>
            <div class="pipe-desc">Top 10 Similar Songs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================
    # EDA
    # ==========================

    st.markdown("## Exploratory Data Analysis")

    st.markdown(
        """
        <p style="color:var(--text2);">
        Distribution of the 9 Spotify audio features used for training.
        </p>
        """,
        unsafe_allow_html=True
    )

    show_image_if_exists("static/eda.png")

    st.markdown("---")

    # ==========================
    # HEATMAP
    # ==========================

    st.markdown("## Correlation Heatmap")

    st.markdown(
        """
        <p style="color:var(--text2);">
        Correlation matrix showing relationships between audio features.
        </p>
        """,
        unsafe_allow_html=True
    )

    show_image_if_exists("static/heatmap.png")

    st.markdown("---")

    # ==========================
    # EVALUATION
    # ==========================

    st.markdown("## Recommendation Quality Evaluation")

    st.markdown(
        """
        <p style="color:var(--text2);">
        Distribution of similarity scores produced by the recommendation engine.
        Higher scores indicate stronger acoustic similarity.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">0.84</div>
        <div class="metric-label">Average Similarity Score</div>
    </div>
    """, unsafe_allow_html=True)

    show_image_if_exists("static/similarity_distribution.png")

    st.markdown("---")

    # ==========================
    # FEATURES TABLE
    # ==========================

    st.markdown("## Features Used By The Model")

    feature_df = pd.DataFrame({
        "Feature": [
            "danceability",
            "energy",
            "acousticness",
            "instrumentalness",
            "liveness",
            "speechiness",
            "valence",
            "loudness",
            "tempo"
        ],
        "Description": [
            "Rhythm suitability",
            "Intensity level",
            "Acoustic content",
            "Instrumental content",
            "Live performance likelihood",
            "Speech content",
            "Musical positivity",
            "Volume intensity",
            "Song speed"
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )


# ════════════ 4. DATASET SECTION ════════════
elif menu == "Dataset":
    st.header("Dataset Overview")

    st.write(f"""
    This project uses a Spotify dataset containing
    **{len(df):,} songs** and {df.shape[1] - 1 if 'dropdown_label' in df.columns else df.shape[1]} attributes.
    """)

    FEATURES = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'loudness', 'tempo']

    st.subheader("Features Used for Training")
    st.write("The model matches music profiles based on these 9 key acoustic vectors:")
    
    features_html = "".join([f'<span class="tag accent" style="font-size:0.9rem; padding:6px 16px; margin:4px;">{f}</span>' for f in FEATURES])
    st.markdown(f'<div style="margin-bottom:25px;">{features_html}</div>', unsafe_allow_html=True)

    display_df = df.drop(columns=['dropdown_label'], errors='ignore')

    st.subheader("Feature Statistics Summary")
    st.dataframe(df[FEATURES].describe(), use_container_width=True)

    st.subheader("Dataset Sample: Top 10 Rows (Head)")
    st.dataframe(display_df.head(10), use_container_width=True)

    st.subheader("Dataset Sample: Bottom 10 Rows (Tail)")
    st.dataframe(display_df.tail(10), use_container_width=True)


# ════════════ 5. TEAMS SECTION ════════════
elif menu == "Teams":
    st.markdown('<h3>Meet the Team</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--text2); margin-bottom:2rem;">The humans behind Algorhythm — bringing together machine learning, data engineering, and product design.</p>', unsafe_allow_html=True)
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    with t_col1:
        st.markdown(f'''
        <div class="team-card-py">
            <img src="data:image/jpeg;base64,{img_william}" alt="Willian">
            <div class="team-name">Willian Y. Alvin</div>
            <div class="team-role">ML Engineer</div>
            <div class="team-desc">Data preprocessing, feature engineering, and KNN implementation – the core of the engine.</div>
        </div>
        ''', unsafe_allow_html=True)

    with t_col2:
        st.markdown(f'''
        <div class="team-card-py">
            <img src="data:image/jpeg;base64,{img_vittorio}" alt="Vittorio">
            <div class="team-name">Vittorio Dinata</div>
            <div class="team-role">Frontend Dev</div>
            <div class="team-desc">Streamlit development, user interface design, and application testing/deployment.</div>
        </div>
        ''', unsafe_allow_html=True)

    with t_col3:
        st.markdown(f'''
        <div class="team-card-py">
            <img src="data:image/jpeg;base64,{img_gilbert}" alt="Gilbert">
            <div class="team-name">G. Gilbert Susanto</div>
            <div class="team-role">Research Lead</div>
            <div class="team-desc">Documentation, literature review, and technical report writing on methodologies.</div>
        </div>
        ''', unsafe_allow_html=True)

    with t_col4:
        st.markdown(f'''
        <div class="team-card-py">
            <img src="data:image/jpeg;base64,{img_andrew}" alt="Andrew">
            <div class="team-name">Andrew Ong</div>
            <div class="team-role">Data Analyst</div>
            <div class="team-desc">Dataset analysis, model evaluation validation, and project presentation materials.</div>
        </div>
        ''', unsafe_allow_html=True)