import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.recommender import AudioRecommender, AUDIO_FEATURES

st.set_page_config(
    page_title="Audio Intelligence & Catalog Engine",
    page_icon="🎵",
    layout="wide"
)

# Friendly mapping for your unsupervised clusters
CLUSTER_LABELS = {
    0: "Electronic / Driving Instrumental",
    1: "Hip-Hop / Vocal Rhythmic",
    2: "Peak Energy / Modern Pop & Rock",
    3: "Ambient / Classical / Deep Focus",
    4: "Acoustic / Stripped Folk & Ballad"
}

@st.cache_resource
def load_engine():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "processed" / "clustered_tracks.parquet"
    models_dir = base_dir / "models"
    recommender = AudioRecommender(data_path, models_dir)
    return recommender

recommender = load_engine()
df = recommender.df

# --- HEADER METRICS ---
st.title("🎵 Music Catalog Audio Intelligence & Recommender")
st.markdown("Unsupervised acoustic profiling, latent-space vector retrieval, and PCA catalog mapping")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Indexed Tracks", f"{len(df):,}")
col2.metric("Acoustic Clusters", f"{df['cluster'].nunique()}")
col3.metric("Catalog Avg BPM", f"{df['tempo'].mean():.1f}")
col4.metric("Avg Dynamic Energy", f"{df['energy'].mean():.2f}")

st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Track Selector")
artist_list = sorted(df["artist_name"].unique())
selected_artist = st.sidebar.selectbox("Filter by Artist:", artist_list)

filtered_tracks = df[df["artist_name"] ==  selected_artist]["track_name"].tolist()
selected_track = st.sidebar.selectbox("Select Track:", filtered_tracks)
top_k = st.sidebar.slider("Recommendations Count:" , min_value=3, max_value=10, value=5)

target_info, recs = recommender.find_similar_tracks(
    selected_track, artist_name=selected_artist, top_n=top_k
)

# Guard against missing target
if target_info is None:
    st.error(f"Could not locate '{selected_track}' by '{selected_artist}'. Please select another track.")
    st.stop()

# --- MAIN DASHBOARD LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📊 Acoustic Radar Profile", "🎯 Sonic Neighbors (Vector Match)", "🗺️ Catalog PCA Map"])

# TAB 1: RADAR CHART
with tab1:
    st.subheader(f"Audio DNA: {target_info['track_name']} by {target_info['artist_name']}")
    assigned_cluster = target_info['cluster']
    st.info(f"**Assigned Archetype:** Cluster {assigned_cluster} - {CLUSTER_LABELS.get(assigned_cluster, 'Custom')}")

    radar_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'valence']
    track_vals = [target_info[f] for f in radar_features]
    catalog_means = [df[f].mean() for f in radar_features]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=track_vals + [track_vals[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name=target_info['track_name'],
        line_color='#1DB954'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=catalog_means + [catalog_means[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name='Catalog Baseline',
        line_color='rgba(150, 150, 150, 0.4)'
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# TAB 2: RECOMMENDATION TABLE
with tab2:
    st.subheader(f"Top {top_k} Closest Acoustic Matches")
    st.caption("Matches retrieved via Cosine Similarity in standardized latent feature space")

    recs["Archetype"] = recs["cluster"].map(CLUSTER_LABELS)
    recs["similarity_pct"] = (recs["similarity_score"] * 100).round(1).astype(str) + "%"

    display_df = recs[[
        "track_name", "artist_name", "Archetype", "similarity_pct", "tempo", "energy", "valence"
    ]].rename(columns={
        "track_name": "Track",
        "artist_name": "Artist",
        "similarity_pct": "Similarity",
        "tempo": "BPM",
        "energy": "Energy",
        "valence": "Valence"
    })
    st.dataframe(display_df, use_container_width=True)

# TAB 3: PCA 2D SCATTER
with tab3:
    st.subheader("2D Latent Catalog Representation (PCA)")
    st.caption("Global distribution of songs across Principal Components with your target track highlighted")

    # Downsample points for fluid rendering
    plot_df = df.sample(n=min(3000, len(df)), random_state=42).copy()
    plot_df["Archetype"] = plot_df["cluster"].map(CLUSTER_LABELS)

    fig_pca = px.scatter(
        plot_df,
        x="pca_x",
        y="pca_y",
        color="Archetype",
        hover_name="track_name",
        hover_data= {
            "artist_name": True,
            "Archetype": True,
            "tempo": ":1f",
            "pca_x": False,
            "pca_y": False
        },
        labels={
            "pca_x": "Principal Component 1 (Acoustic Density & Energy)",
            "pca_y": "Principal Component 2 (Vocal Presence & Valence)",
            "artist_name": "Artist",
            "tempo": "BPM",
        },
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    # Add target track marker
    fig_pca.add_trace(go.Scatter(
        x=[target_info["pca_x"]],
        y=[target_info["pca_y"]],
        mode="markers+text",
        marker=dict(color="red", size=14, symbol="star", line=dict(width=2, color="white")),
        name="Selected Song",
        text=[target_info["track_name"]],
        textposition="top center",
        hovertemplate=(
            f"<b>{target_info['track_name']}</b><br>"
            f"Artist: {target_info['artist_name']}<br>"
            f"Archetype: {CLUSTER_LABELS.get(target_info['cluster'], 'Custom')}<br>"
            f"BPM: {target_info['tempo']:.1f}<extra></extra>"
        )
    ))
    fig_pca.update_layout(
        height=580,
        legend_title_text="Acoustic Archetype",
        xaxis=dict(gridcolor="rgba(240, 240, 240, 0.6)"),
        yaxis=dict(gridcolor="rgba(240, 240, 240, 0.6)")
    )
    st.plotly_chart(fig_pca, use_container_width=True)





















