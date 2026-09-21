from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "valence", "tempo"
]

class AudioRecommender:
    def __init__(self, data_path: Path, models_dir: Path):
        self.df = pd.read_parquet(data_path)
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.kmeans = joblib.load(models_dir / "kmeans.joblib")

        # Precompute normalized feature matrix for fast similarity lookup
        self.feature_matrix = self.scaler.transform(self.df[AUDIO_FEATURES])

    def find_similar_tracks(self, track_name: str, artist_name: str = None, top_n: int = 5):
        """
        Locates a track by name/artist, computes cosine similarity
        against all tracks in the catalog, and returns the closest match
        """

        # Exact matching is faster and immune to special characters / regex errors
        if artist_name:
            matches = self.df[(self.df["track_name"] == track_name) & (self.df["artist_name"] == artist_name)]
        else:
            matches = self.df[self.df["track_name"] == track_name]

        # Fallback to contains if exact match didn't find anything
        if matches.empty:
            matches = self.df[self.df["track_name"].str.contains(track_name, case=False, regex=False, na=False)]

        if matches.empty:
            return None, pd.DataFrame()

        # Take first match if multiple exist
        target_idx = matches.index[0]
        target_vector = self.feature_matrix[target_idx].reshape(1, -1)

        # Compute cosine similarity across the pre-scaled feature matrix
        similarities = cosine_similarity(target_vector, self.feature_matrix).flatten()

        # Retrieve top indices (excluding the target track itself)
        sorted_indices = similarities.argsort()[::-1]
        top_indices = [idx for idx in sorted_indices if idx != target_idx][:top_n]

        results = self.df.iloc[top_indices][
            ["track_name", "artist_name", "cluster", "tempo", "energy", "valence"]
        ].copy()
        results["similarity_score"] = similarities[top_indices].round(4)

        target_info = self.df.iloc[target_idx]
        return target_info, results.reset_index(drop=True)

if __name__ == "__main__":
    PROJECT_ROOT =Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / "data" / "processed" / "clustered_tracks.parquet"
    models_dir = PROJECT_ROOT / "models"

    recommender = AudioRecommender(data_path, models_dir)

    # Test with the first track in the dataset
    sample_track = recommender.df.iloc[0]["track_name"]
    sample_artist = recommender.df.iloc[0]["artist_name"]
    print(f"Testing recommendations for: '{sample_track}' by '{sample_artist}\n")

    target, recs = recommender.find_similar_tracks(sample_track, sample_artist, top_n=5)
    print("Target Track Cluster:", target["cluster"])
    print("\nTop 5 Sonic Neighbors:")
    print(recs)

