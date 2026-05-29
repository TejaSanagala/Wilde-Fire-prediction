"""Reusable training and analysis pipeline for the California wildfire project.

This script is intentionally separate from the original notebook so the project
can be run as a normal GitHub portfolio project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "CA_Weather_Fire_Dataset_1984-2025.csv"
DEFAULT_INCIDENT_PATH = PROJECT_ROOT / "data" / "raw" / "California_Fire_Incidents.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


NUMERIC_WEATHER_COLUMNS = [
    "PRECIPITATION",
    "MAX_TEMP",
    "MIN_TEMP",
    "AVG_WIND_SPEED",
    "TEMP_RANGE",
    "WIND_TEMP_RATIO",
]

INCIDENT_COLUMNS = [
    "AcresBurned",
    "Latitude",
    "Longitude",
    "Started",
    "Status",
    "Counties",
]


def load_datasets(weather_path: Path, incident_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load weather and fire incident datasets from local CSV files."""
    import pandas as pd

    missing_paths = [path for path in [weather_path, incident_path] if not path.exists()]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Required dataset file(s) were not found:\n"
            f"{missing}\n\n"
            "Download the datasets listed in data/README.md and place them in data/raw/."
        )

    weather = pd.read_csv(weather_path)
    incidents = pd.read_csv(incident_path)
    return weather, incidents


def clean_and_merge(weather: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    """Clean source datasets and merge them by fire start date."""
    import pandas as pd

    weather = weather.copy()
    incidents = incidents.copy()

    for column in NUMERIC_WEATHER_COLUMNS:
        if column in weather.columns:
            weather[column] = weather[column].fillna(weather[column].mean())

    incidents = incidents[INCIDENT_COLUMNS].copy()
    incidents["AcresBurned"] = incidents["AcresBurned"].fillna(incidents["AcresBurned"].mean())

    weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce").dt.date
    incidents["Started"] = pd.to_datetime(incidents["Started"], errors="coerce").dt.date
    incidents = incidents.dropna(subset=["Started"])

    merged = pd.merge(weather, incidents, left_on="DATE", right_on="Started", how="inner")

    selected_columns = [
        "DATE",
        "PRECIPITATION",
        "MAX_TEMP",
        "MIN_TEMP",
        "AVG_WIND_SPEED",
        "FIRE_START_DAY",
        "YEAR",
        "TEMP_RANGE",
        "WIND_TEMP_RATIO",
        "MONTH",
        "SEASON",
        "LAGGED_PRECIPITATION",
        "LAGGED_AVG_WIND_SPEED",
        "DAY_OF_YEAR",
        "AcresBurned",
        "Latitude",
        "Longitude",
        "Started",
        "Status",
        "Counties",
    ]
    available_columns = [column for column in selected_columns if column in merged.columns]
    return merged[available_columns].copy()


def save_visualizations(data: pd.DataFrame, figures_dir: Path) -> None:
    """Create the main exploratory visualizations from the project."""
    import matplotlib.pyplot as plt
    import plotly.express as px
    import seaborn as sns

    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fire_per_year = data.groupby("YEAR")["FIRE_START_DAY"].sum()
    plt.figure(figsize=(12, 6))
    fire_per_year.plot(kind="bar", color="#e57200")
    plt.xlabel("Year")
    plt.ylabel("Total Fire Start Days")
    plt.title("Total Wildfire Occurrences per Year")
    plt.tight_layout()
    plt.savefig(figures_dir / "wildfires_per_year.png", dpi=180)
    plt.close()

    fire_by_season = data.groupby("SEASON")["FIRE_START_DAY"].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=fire_by_season.index, y=fire_by_season.values, palette="flare", hue=fire_by_season.index, legend=False)
    plt.xlabel("Season")
    plt.ylabel("Total Fire Start Days")
    plt.title("Total Wildfire Start Days by Season")
    plt.tight_layout()
    plt.savefig(figures_dir / "wildfires_by_season.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x="FIRE_START_DAY", y="PRECIPITATION")
    plt.xlabel("Fire Start Day")
    plt.ylabel("Precipitation")
    plt.title("Distribution of Precipitation on Fire vs Non-Fire Days")
    plt.tight_layout()
    plt.savefig(figures_dir / "precipitation_fire_vs_non_fire.png", dpi=180)
    plt.close()

    county_acres = data.groupby("Counties")["AcresBurned"].sum().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=county_acres.values, y=county_acres.index, color="#5f8d4e")
    plt.xlabel("Total Acres Burned")
    plt.ylabel("County")
    plt.title("Top Counties by Total Acres Burned")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_counties_by_acres_burned.png", dpi=180)
    plt.close()

    map_data = data.dropna(subset=["Latitude", "Longitude", "AcresBurned"]).copy()
    if not map_data.empty:
        fig = px.scatter_mapbox(
            map_data,
            lat="Latitude",
            lon="Longitude",
            size="AcresBurned",
            color="AcresBurned",
            color_continuous_scale="YlOrRd",
            size_max=18,
            zoom=5,
            hover_name="Counties",
            hover_data={"AcresBurned": True, "Latitude": False, "Longitude": False},
            title="California Wildfire Acres Burned by Location",
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": 36.7783, "lon": -119.4179},
            margin={"r": 0, "t": 45, "l": 0, "b": 0},
        )
        fig.write_html(figures_dir / "wildfire_location_map.html")


def prepare_model_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model-ready features and target variable."""
    import pandas as pd

    model_data = data.dropna(subset=["FIRE_START_DAY"]).copy()
    model_data = pd.get_dummies(model_data, columns=["SEASON"], drop_first=True)
    model_data = model_data.drop(columns=["DATE", "Started"], errors="ignore")

    y = model_data["FIRE_START_DAY"].astype(int)
    X = model_data.drop(columns=["FIRE_START_DAY"], errors="ignore")

    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.fillna(X.median(numeric_only=True))
    return X, y


def train_classifier(data: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    """Train and evaluate a Random Forest wildfire classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    X, y = prepare_model_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "model_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "Random Forest Classifier Results",
                "================================",
                f"Accuracy: {accuracy:.4f}",
                f"ROC AUC: {roc_auc:.4f}",
                "",
                "Confusion Matrix:",
                str(confusion_matrix(y_test, predictions)),
                "",
                "Classification Report:",
                classification_report(y_test, predictions),
            ]
        ),
        encoding="utf-8",
    )

    return {"accuracy": accuracy, "roc_auc": roc_auc}


def run_clustering(data: pd.DataFrame, output_dir: Path, n_clusters: int = 3) -> float:
    """Apply K-Means clustering and save clustered wildfire records."""
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cluster_data = data.copy()
    cluster_data = pd.get_dummies(cluster_data, columns=["SEASON"], drop_first=True)
    cluster_data = cluster_data.drop(columns=["DATE", "Started"], errors="ignore")

    for column in cluster_data.select_dtypes(include=["object"]).columns:
        cluster_data[column] = pd.to_numeric(cluster_data[column], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(cluster_data)
    X_scaled = scaler.fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    clustered = data.copy()
    clustered["Cluster"] = clusters
    clustered.to_csv(output_dir / "clustered_wildfire_records.csv", index=False)

    return float(kmeans.inertia_)


def run_pipeline(weather_path: Path, incident_path: Path, output_dir: Path) -> None:
    """Run the full data cleaning, visualization, classification, and clustering flow."""
    weather, incidents = load_datasets(weather_path, incident_path)
    data = clean_and_merge(weather, incidents)

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "merged_wildfire_dataset.csv", index=False)

    save_visualizations(data, output_dir / "figures")
    metrics = train_classifier(data, output_dir)
    inertia = run_clustering(data, output_dir)

    print("Pipeline completed successfully.")
    print(f"Merged rows: {len(data):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"K-Means inertia: {inertia:.2f}")
    print(f"Outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the California wildfire ML pipeline.")
    parser.add_argument("--weather-data", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--incident-data", type=Path, default=DEFAULT_INCIDENT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.weather_data, args.incident_data, args.output_dir)
