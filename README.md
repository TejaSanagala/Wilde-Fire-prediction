# California Wildfire Prediction and Risk Analysis

This project analyzes California wildfire patterns using weather observations, historical fire incidents, and machine learning. The goal is to identify conditions associated with wildfire starts and highlight geographic areas with higher wildfire impact.

The repository has been organized as a portfolio-ready machine learning project with a reproducible Python pipeline, documented datasets, saved outputs, and the original exploratory notebook.

## Project Highlights

- Cleaned and merged weather and wildfire incident datasets.
- Explored wildfire trends by year, season, precipitation, county, and location.
- Built a Random Forest classifier to predict wildfire start days.
- Applied K-Means clustering to group wildfire records by weather, geographic, and burn-severity patterns.
- Documented limitations around class imbalance, dataset alignment, and cluster separation.

## Problem Statement

Wildfires are a major environmental and public safety risk in California. By combining historical wildfire incidents with weather conditions such as precipitation, temperature, wind speed, and seasonal indicators, this project investigates whether machine learning can help identify wildfire risk patterns and support more informed prevention and response planning.

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   └── raw/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── mlpp25_project_steja.ipynb
│   └── mlpp25_project_steja.py
├── outputs/
│   └── figures/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   └── wildfire_pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Datasets

This project uses two public datasets:

1. California Weather and Fire Prediction Dataset, 1984-2025  
   Source: https://zenodo.org/records/14712845

2. California Wildfire Incidents, 2013-2020  
   Source: https://www.kaggle.com/datasets/ananthu017/california-wildfire-incidents-20132020

The raw CSV files are not committed to this repository. Download them from the links above and place them in `data/raw/`:

```text
data/raw/CA_Weather_Fire_Dataset_1984-2025.csv
data/raw/California_Fire_Incidents.csv
```

See [data/README.md](data/README.md) for more details.

## Methods

### Data Preparation

- Loaded weather and wildfire incident records.
- Filled missing numeric values using mean imputation for the original analysis.
- Converted date fields to a common format.
- Merged the datasets using weather observation date and wildfire start date.
- Selected weather, seasonal, geographic, and burn-impact features.

### Exploratory Analysis

The project includes visual analysis of:

- Wildfire start days by year.
- Wildfire start days by season.
- Precipitation differences between fire and non-fire days.
- Counties with the highest total acres burned.
- Geographic wildfire distribution using latitude and longitude.

### Classification

A Random Forest classifier was used to predict wildfire start days from weather and engineered features. The original notebook reported:

- Accuracy: approximately 92%
- ROC AUC: approximately 0.90

The analysis also found that class imbalance affected model reliability. The model performed strongly on wildfire days but struggled to identify non-fire days, which is an important limitation for real-world use.

### Clustering

K-Means clustering was used to identify groups of wildfire records based on weather, location, and acres burned. The original analysis used three clusters and reported a silhouette score around 0.305, suggesting weak to moderate separation between clusters.

## How to Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Add the raw datasets to `data/raw/`, then run:

```powershell
python src/wildfire_pipeline.py
```

Optional custom paths:

```powershell
python src/wildfire_pipeline.py `
  --weather-data data/raw/CA_Weather_Fire_Dataset_1984-2025.csv `
  --incident-data data/raw/California_Fire_Incidents.csv `
  --output-dir outputs
```

Generated artifacts are saved under `outputs/`, including:

- `merged_wildfire_dataset.csv`
- `model_report.txt`
- `clustered_wildfire_records.csv`
- visualization files in `outputs/figures/`

## Results Summary

The project demonstrates that weather and incident data can reveal meaningful wildfire risk patterns. Seasonal trends showed higher fire activity during warmer and drier periods, and geographic visualizations highlighted counties and regions with larger burned areas.

The classification model achieved strong headline accuracy, but the detailed metrics showed class imbalance issues. This means the result should be interpreted carefully: high accuracy alone does not guarantee reliable wildfire risk prediction.

## Limitations and Future Work

- Raw datasets have different temporal coverage, which limits perfect alignment.
- Mean imputation may smooth meaningful weather extremes.
- The classification task is affected by class imbalance.
- K-Means clustering showed only weak to moderate separation.
- Future improvements could include resampling methods, threshold tuning, additional weather features, geospatial features, and model comparison with gradient boosting or time-aware validation.

## Resume-Friendly Summary

Built a California wildfire risk analysis project using Python, pandas, scikit-learn, seaborn, and Plotly. Cleaned and merged weather and wildfire incident datasets, engineered model-ready features, trained a Random Forest classifier for wildfire start prediction, and applied K-Means clustering to identify geographic wildfire risk patterns. Documented model performance, class imbalance limitations, and opportunities for future improvement.

## Tools Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Plotly
- Jupyter Notebook
