# Data Instructions

This repository does not include the raw datasets because they are externally sourced and may be large.

Place the following files in `data/raw/` before running the pipeline:

1. `CA_Weather_Fire_Dataset_1984-2025.csv`
2. `California_Fire_Incidents.csv`

Dataset sources:

- California Weather and Fire Prediction Dataset, 1984-2025: https://zenodo.org/records/14712845
- California Wildfire Incidents, 2013-2020: https://www.kaggle.com/datasets/ananthu017/california-wildfire-incidents-20132020

Expected directory layout:

```text
data/
  raw/
    CA_Weather_Fire_Dataset_1984-2025.csv
    California_Fire_Incidents.csv
```

The pipeline reads from `data/raw/` and writes cleaned data, reports, and figures to `outputs/`.
