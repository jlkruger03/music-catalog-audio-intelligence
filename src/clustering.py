from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# Core continuous audio attributions
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "valence", "tempo"
]

def build_clustering_pipeline(
        input_data_path: Path,
        output_data_path: Path,
        models_dir: Path,
        n_clusters: int = 5,
        random_state: int = 42
):
    """
    Fits StandardScaler, runs PCA for dimensionality reduction,
    trains a K-Means clustering model, and saves artifacts for inference.
    """
    print(f"Loading curated data from: {input_data_path}...")
    df = pd.read_parquet(input_data_path)
    print(f"Dataset shape: {df.shape}")

    # 1. Feature Scaling
    # Raw features vary widely (e.g. tempo ~120 vs. valence ~0.5), so scaling is required
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[AUDIO_FEATURES])

    # 2. Dimensionality Reduction (PCA for 2D visual mapping)
    print("Fitting PCA (2 components)...")
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]

    explained_var = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA 2-Component Explained Variance: {explained_var:.2f}%")

    # 3. K-Means Acoustic Mood Clustering
    print(f"Training KMeans model (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # 4. Profile Cluster Archetypes
    print("\n--- Acoustic Cluster Profiles (Centroid Means) ---")
    cluster_profiles = df.groupby("cluster")[AUDIO_FEATURES].mean().round(3)
    print(cluster_profiles)

    # 5. Export Processed Data & Model Artifacts
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save augmented parquet dataset with cluster & PCA labels
    df.to_parquet(output_data_path, index=False)
    print(f"\nSaved clustered dataset to: {output_data_path}")

    # Serialize scaler and models using joblib
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(pca, models_dir / "pca.joblib")
    joblib.dump(kmeans, models_dir / "kmeans.joblib")
    print(f"Successfully exported models to: {models_dir}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    input_data_path = PROJECT_ROOT / "data" / "processed" / "clean_tracks.parquet"
    output_data_path = PROJECT_ROOT / "data" / "processed" / "clustered_tracks.parquet"
    models_directory = PROJECT_ROOT / "models"

    build_clustering_pipeline(input_data_path, output_data_path, models_directory)