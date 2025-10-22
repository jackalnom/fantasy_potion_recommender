"""
Visualize Collaborative Filtering Embeddings using UMAP and Plotly.

This script:
1. Trains the Surprise SVD recommender on the fantasy potion data
2. Extracts user (adventurer) and item (potion) embeddings
3. Reduces dimensionality to 3D using UMAP
4. Creates interactive 3D Plotly visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from recommender_models.surprise_recommender import SurpriseRecommender
from data_utils import RecommenderDataPrep, CF_FEATURE_COLS


def extract_embeddings(model):
    """Extract user and item embeddings from trained Surprise SVD model.

    Args:
        model: Trained SurpriseRecommender instance

    Returns:
        user_embeddings: np.array of shape (n_users, n_factors)
        item_embeddings: np.array of shape (n_items, n_factors)
        user_ids: list of original user IDs
        item_ids: list of original item IDs
    """
    # Extract latent factors from the SVD model
    # Surprise stores these as pu (user factors) and qi (item factors)
    trainset = model.model.trainset

    # Get the number of users and items
    n_users = trainset.n_users
    n_items = trainset.n_items

    # Extract embeddings
    user_embeddings = np.array([model.model.pu[i] for i in range(n_users)])
    item_embeddings = np.array([model.model.qi[i] for i in range(n_items)])

    # Get original IDs (map from inner IDs back to raw IDs)
    user_ids = [trainset.to_raw_uid(i) for i in range(n_users)]
    item_ids = [trainset.to_raw_iid(i) for i in range(n_items)]

    return user_embeddings, item_embeddings, user_ids, item_ids


def reduce_to_3d(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """Reduce embeddings to 3D using UMAP.

    Args:
        embeddings: np.array of shape (n_samples, n_features)
        n_neighbors: UMAP parameter controlling local vs global structure
        min_dist: UMAP parameter controlling clustering tightness
        random_state: Random seed for reproducibility

    Returns:
        embeddings_3d: np.array of shape (n_samples, 3)
    """
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='cosine'
    )
    embeddings_3d = reducer.fit_transform(embeddings)
    return embeddings_3d


def create_user_visualization(embeddings_3d, user_ids, adv_info):
    """Create 3D scatter plot for user embeddings colored by class.

    Args:
        embeddings_3d: np.array of shape (n_users, 3)
        user_ids: list of user IDs
        adv_info: DataFrame with columns ['class', 'level'] indexed by adv_id

    Returns:
        plotly Figure object
    """
    # Get class and level info for each user
    classes = [adv_info.loc[uid, 'class'] for uid in user_ids]
    levels = [adv_info.loc[uid, 'level'] for uid in user_ids]

    # Create color mapping for classes
    unique_classes = sorted(set(classes))
    color_map = {cls: i for i, cls in enumerate(unique_classes)}
    colors = [color_map[cls] for cls in classes]

    # Create hover text
    hover_text = [
        f"Adventurer {uid}<br>Class: {cls}<br>Level: {lvl}"
        for uid, cls, lvl in zip(user_ids, classes, levels)
    ]

    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Class",
                tickvals=list(range(len(unique_classes))),
                ticktext=unique_classes
            ),
            line=dict(width=0.5, color='white')
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    )])

    fig.update_layout(
        title='Adventurer Embeddings (UMAP 3D)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1000,
        height=800
    )

    return fig


def normalize_rgb_to_max(r, g, b):
    """Normalize RGB values so the max component becomes 255.

    For example: (50, 50, 0) -> (255, 255, 0) bright yellow
    This makes colors brighter and more saturated for visualization.
    """
    max_val = max(r, g, b)
    if max_val == 0:
        return 0, 0, 0

    # Scale so max component is 255
    scale = 255 / max_val
    return int(r * scale), int(g * scale), int(b * scale)


def create_item_visualization(embeddings_3d, item_ids, potion_features):
    """Create 3D scatter plot for item embeddings colored by actual potion RGB colors.

    Args:
        embeddings_3d: np.array of shape (n_items, 3)
        item_ids: list of item IDs
        potion_features: DataFrame with columns ['red', 'green', 'blue'] indexed by potion_id

    Returns:
        plotly Figure object
    """
    # Convert RGB values to hex colors with normalization
    hex_colors = []
    rgb_values = []
    rgb_normalized = []
    for iid in item_ids:
        r_raw = int(potion_features.loc[iid, 'red'])
        g_raw = int(potion_features.loc[iid, 'green'])
        b_raw = int(potion_features.loc[iid, 'blue'])
        rgb_values.append((r_raw, g_raw, b_raw))

        # Normalize to make colors brighter
        r_norm, g_norm, b_norm = normalize_rgb_to_max(r_raw, g_raw, b_raw)
        rgb_normalized.append((r_norm, g_norm, b_norm))
        hex_colors.append(f'#{r_norm:02x}{g_norm:02x}{b_norm:02x}')

    # Create hover text
    hover_text = [
        f"Potion {iid}<br>Raw RGB: ({r_raw}, {g_raw}, {b_raw})<br>"
        f"Normalized: ({r_norm}, {g_norm}, {b_norm})<br>Hex: {hex_color}"
        for iid, (r_raw, g_raw, b_raw), (r_norm, g_norm, b_norm), hex_color
        in zip(item_ids, rgb_values, rgb_normalized, hex_colors)
    ]

    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=hex_colors,
            line=dict(width=0.5, color='white')
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    )])

    fig.update_layout(
        title='Potion Embeddings (UMAP 3D) - Colored by RGB Values',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1000,
        height=800
    )

    return fig


def create_combined_visualization(user_embeddings_3d, item_embeddings_3d,
                                   user_ids, item_ids, adv_info, potion_features):
    """Create combined 3D scatter plot showing both users and items.

    Args:
        user_embeddings_3d: np.array of shape (n_users, 3)
        item_embeddings_3d: np.array of shape (n_items, 3)
        user_ids: list of user IDs
        item_ids: list of item IDs
        adv_info: DataFrame with adventurer info
        potion_features: DataFrame with potion features

    Returns:
        plotly Figure object
    """
    # User trace
    classes = [adv_info.loc[uid, 'class'] for uid in user_ids]
    levels = [adv_info.loc[uid, 'level'] for uid in user_ids]
    user_hover = [
        f"Adventurer {uid}<br>Class: {cls}<br>Level: {lvl}"
        for uid, cls, lvl in zip(user_ids, classes, levels)
    ]

    user_trace = go.Scatter3d(
        x=user_embeddings_3d[:, 0],
        y=user_embeddings_3d[:, 1],
        z=user_embeddings_3d[:, 2],
        mode='markers',
        name='Adventurers',
        marker=dict(
            size=5,
            color='lightblue',
            symbol='circle',
            line=dict(width=0.5, color='darkblue'),
            opacity=0.6
        ),
        text=user_hover,
        hovertemplate='%{text}<extra></extra>'
    )

    # Item trace colored by actual RGB values with normalization
    hex_colors = []
    rgb_values = []
    rgb_normalized = []
    for iid in item_ids:
        r_raw = int(potion_features.loc[iid, 'red'])
        g_raw = int(potion_features.loc[iid, 'green'])
        b_raw = int(potion_features.loc[iid, 'blue'])
        rgb_values.append((r_raw, g_raw, b_raw))

        # Normalize to make colors brighter
        r_norm, g_norm, b_norm = normalize_rgb_to_max(r_raw, g_raw, b_raw)
        rgb_normalized.append((r_norm, g_norm, b_norm))
        hex_colors.append(f'#{r_norm:02x}{g_norm:02x}{b_norm:02x}')

    item_hover = [
        f"Potion {iid}<br>Raw RGB: ({r_raw}, {g_raw}, {b_raw})<br>"
        f"Normalized: ({r_norm}, {g_norm}, {b_norm})<br>Hex: {hex_color}"
        for iid, (r_raw, g_raw, b_raw), (r_norm, g_norm, b_norm), hex_color
        in zip(item_ids, rgb_values, rgb_normalized, hex_colors)
    ]

    item_trace = go.Scatter3d(
        x=item_embeddings_3d[:, 0],
        y=item_embeddings_3d[:, 1],
        z=item_embeddings_3d[:, 2],
        mode='markers',
        name='Potions',
        marker=dict(
            size=8,
            color=hex_colors,
            symbol='diamond',
            line=dict(width=1, color='white')
        ),
        text=item_hover,
        hovertemplate='%{text}<extra></extra>'
    )

    fig = go.Figure(data=[user_trace, item_trace])

    fig.update_layout(
        title='Combined Embeddings: Adventurers & Potions (UMAP 3D)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1200,
        height=900,
        showlegend=True
    )

    return fig


def main():
    """Main execution function."""
    print("Loading and preparing data...")
    data_prep = RecommenderDataPrep()
    data_prep.load_and_prepare()
    X_train, y_train = data_prep.get_training_data(CF_FEATURE_COLS)

    print(f"Training data shape: {X_train.shape}")
    print(f"Number of unique users: {X_train['adv_id'].nunique()}")
    print(f"Number of unique items: {X_train['potion_id'].nunique()}")

    # Train the Surprise recommender
    print("\nTraining Surprise SVD recommender...")
    recommender = SurpriseRecommender(n_factors=50, n_epochs=50)
    recommender.fit(X_train, y_train)
    print("Training complete!")

    # Extract embeddings
    print("\nExtracting embeddings from trained model...")
    user_embeddings, item_embeddings, user_ids, item_ids = extract_embeddings(recommender)

    print(f"User embeddings shape: {user_embeddings.shape}")
    print(f"Item embeddings shape: {item_embeddings.shape}")

    # Reduce to 3D using UMAP
    print("\nReducing user embeddings to 3D with UMAP...")
    user_embeddings_3d = reduce_to_3d(user_embeddings)

    print("Reducing item embeddings to 3D with UMAP...")
    item_embeddings_3d = reduce_to_3d(item_embeddings)

    # Create visualizations
    print("\nCreating visualizations...")

    print("1. User embeddings visualization...")
    user_fig = create_user_visualization(
        user_embeddings_3d,
        user_ids,
        data_prep.adv_info
    )
    user_fig.write_html("user_embeddings_3d.html")
    print("   Saved to: user_embeddings_3d.html")

    print("2. Item embeddings visualization (colored by RGB values)...")
    item_fig = create_item_visualization(
        item_embeddings_3d,
        item_ids,
        data_prep.potion_features
    )
    item_fig.write_html("item_embeddings_3d.html")
    print("   Saved to: item_embeddings_3d.html")

    print("3. Combined embeddings visualization...")
    combined_fig = create_combined_visualization(
        user_embeddings_3d,
        item_embeddings_3d,
        user_ids,
        item_ids,
        data_prep.adv_info,
        data_prep.potion_features
    )
    combined_fig.write_html("combined_embeddings_3d.html")
    print("   Saved to: combined_embeddings_3d.html")

    print("\nAll visualizations created successfully!")
    print("\nOpen the HTML files in your browser to explore the interactive 3D plots.")

    # Optional: Save embeddings to CSV for further analysis
    print("\nSaving embeddings to CSV files...")
    user_df = pd.DataFrame(
        user_embeddings_3d,
        columns=['umap_1', 'umap_2', 'umap_3']
    )
    user_df.insert(0, 'adv_id', user_ids)
    user_df.to_csv("user_embeddings_3d.csv", index=False)

    item_df = pd.DataFrame(
        item_embeddings_3d,
        columns=['umap_1', 'umap_2', 'umap_3']
    )
    item_df.insert(0, 'potion_id', item_ids)
    item_df.to_csv("item_embeddings_3d.csv", index=False)

    print("   Saved embeddings to: user_embeddings_3d.csv, item_embeddings_3d.csv")


if __name__ == "__main__":
    main()
