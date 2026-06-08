import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle

def load_train_and_save_pipeline():
    df = pd.read_csv("data.csv")
    
    #clean
    title_col = 'name' if 'name' in df.columns else 'title'
    if title_col in df.columns and 'artists' in df.columns:
        initial_shape = df.shape[0]
        df = df.drop_duplicates(subset=[title_col, 'artists'], keep='first')
        df = df.reset_index(drop=True)
        print(f"Delete {initial_shape - df.shape[0]} Duplicates.")

    if 'artists' in df.columns and df['artists'].astype(str).str.startswith('[').any():
        df['artists'] = df['artists'].astype(str).str.replace(r"\[|\]|'", "", regex=True)
        
    #scaling
    features = [
        'danceability', 'energy', 'acousticness', 'instrumentalness',
        'liveness', 'speechiness', 'valence', 'loudness', 'tempo'
    ]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #train
    model = NearestNeighbors(
        n_neighbors=11,
        metric='cosine',
        algorithm='brute'
    )
    model.fit(X_scaled)

    pipeline_data = {
        'model': model,
        'X_scaled': X_scaled,
        'df': df
    }
    with open("music_recommender.pkl", "wb") as f:
        pickle.dump(pipeline_data, f)

if __name__ == "__main__":
    print("Start")
    load_train_and_save_pipeline()
    print("OK")    