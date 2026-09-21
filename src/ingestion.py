from pathlib import Path

import pandas as pd


def clean_spotify_catalog(
    raw_path,
    output_path,
    sample_size: int = 50000,
    max_loudness_db: float = 0.0,
):
    """
    Ingest raw music metadata, handle schema variations, clean audio
    signal anomalies, remove duplicates, and save to Parquet.
    """
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {raw_path.resolve()}\n"
            f"Current working directory: {Path.cwd()}"
        )

    print(f"Reading raw data from: {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Initial raw record count: {len(df):,}")

    # Standardize column names to lowercase and strip whitespace
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Map common column aliases across Kaggle Spotify versions
    rename_mapping = {
        "artists": "artist_name",
        "artist": "artist_name",
        "name": "track_name",
        "track": "track_name",
        "track_genre": "genre",
    }
    # Only rename when the target name doesn't already exist (avoids duplicate columns)
    rename_mapping = {
        old: new
        for old, new in rename_mapping.items()
        if old in df.columns and new not in df.columns
    }
    df = df.rename(columns=rename_mapping)

    # Core required columns
    text_cols = ["track_name", "artist_name"]
    audio_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "valence", "tempo",
    ]
    required_cols = text_cols + audio_cols

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing essential columns: {missing_cols}")

    # Strip brackets, single quotes, double quotes and trailing whitespace from artist names
    df["artist_name"] = (
        df["artist_name"]
        .astype(str)
        .str.strip("[]'\" ")
    )

    # Coerce audio features to numeric; bad values become NaN and get dropped below
    df[audio_cols] = df[audio_cols].apply(pd.to_numeric, errors="coerce")

    # Drop records missing critical metadata or audio values
    df = df.dropna(subset=required_cols)

    # Remove duplicates on identical track and artist combinations
    before = len(df)
    df = df.drop_duplicates(subset=text_cols)
    print(f"Duplicates removed: {before - len(df):,}")

    # Audio signal sanitization
    # 1. Non-positive tempo is physically impossible
    df = df[df["tempo"] > 0]

    # 2. Loudness above the ceiling (default 0 dB) is treated as invalid
    before = len(df)
    df = df[df["loudness"] <= max_loudness_db]
    print(f"Rows removed by loudness filter: {before - len(df):,}")

    # 3. Probabilistic audio features must lie within [0.0, 1.0]
    prob_features = [
        "danceability", "energy", "valence",
        "speechiness", "acousticness", "instrumentalness",
    ]
    for col in prob_features:
        df = df[(df[col] >= 0.0) & (df[col] <= 1.0)]

    print(f"Valid records after sanitization: {len(df):,}")

    # Sample for fast, responsive ML inference in Streamlit
    if len(df) > sample_size:
        print(f"Subsampling to {sample_size:,} tracks for dashboard performance...")
        df = df.sample(n=sample_size, random_state=42)

    df = df.reset_index(drop=True)

    # Ensure the output directory exists, then export to Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    print(f"Curated dataset successfully saved to: {output_path}")
    print(f"Final output shape: {df.shape}")


if __name__ == "__main__":
    # ingestion.py is in src/, so go up two levels to reach the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tracks.csv"
    OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_tracks.parquet"

    print("Looking for:", RAW_DATA_PATH)
    clean_spotify_catalog(RAW_DATA_PATH, OUTPUT_DATA_PATH)