from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROUTE_COLUMNS = [
    "route_25_latitude", "route_25_longitude",
    "route_50_latitude", "route_50_longitude",
    "route_75_latitude", "route_75_longitude",
]


def assign_corridor_choices(frame: pd.DataFrame) -> pd.Series:
    """Assign stable, geography-ordered corridor labels to recorded journeys."""
    labels = pd.Series("single corridor", index=frame.index, dtype="object")
    if len(frame) < 6 or not set(ROUTE_COLUMNS).issubset(frame):
        return labels
    routes = frame[ROUTE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = routes.notna().all(axis=1)
    if valid.sum() < 6 or routes.loc[valid].var().sum() <= 0:
        return labels
    scaled = StandardScaler().fit_transform(routes.loc[valid])
    clusters = KMeans(n_clusters=2, random_state=41, n_init=20).fit_predict(scaled)
    counts = pd.Series(clusters).value_counts()
    if counts.min() < 2 or silhouette_score(scaled, clusters) < 0.15:
        return labels
    centers = frame.loc[valid].assign(cluster=clusters).groupby("cluster")[[
        "route_50_longitude", "route_50_latitude"
    ]].mean()
    ordered = centers.sort_values(["route_50_longitude", "route_50_latitude"]).index
    label_map = {
        int(cluster): f"corridor {position + 1}"
        for position, cluster in enumerate(ordered)
    }
    labels.loc[valid] = pd.Series(clusters, index=frame.index[valid]).map(label_map)
    return labels
