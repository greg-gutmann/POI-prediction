# Datasets

This document summarizes the datasets used, the required file formats, and how to plug in additional datasets.

## 1) Datasets used in this repo

- **Foursquare (TSMC 2014 Tokyo)**
  - Source: https://sites.google.com/site/yangdingqi/home/foursquare-dataset
  - Expected raw file: `data/tky/raw/dataset_TSMC2014_TKY.txt`
  - Reader: TSV (tab-separated), schema below.

- **Gowalla (California & Nevada subset)**
  - Source (original): https://snap.stanford.edu/data/loc-Gowalla.html
  - This repo uses a CA+NV subset generated from the original Gowalla check-ins and a spot subset file.
  - Script: `generate_ca_raw.py` builds `data/ca/raw/dataset_gowalla_ca_ne.csv` from:
    - `data/ca/raw/loc-gowalla_totalCheckins.txt`
    - `data/ca/raw/gowalla_spots_subset1.csv`
    - `data/ca/raw/us_state_polygon_json.json` (polygons to filter CA/NV)

- **NYC (pre-split CSVs)**
  - Provided via the project data bundle (see README data link).
  - Expected in: `data/nyc/raw/`
  - Files: `NYC_train.csv`, `NYC_val.csv`, `NYC_test.csv`

## 2) Directory layout

```
/data
  /nyc
    /raw
      NYC_train.csv
      NYC_val.csv
      NYC_test.csv
    /preprocessed
      sample.csv
      train_sample.csv
      validate_sample.csv
      test_sample.csv
      label_encoding.pkl
      ci2traj_pyg_data.pt
      traj2traj_pyg_data.pt
  /tky
    /raw
      dataset_TSMC2014_TKY.txt
    /preprocessed
      ... (same as above)
  /ca
    /raw
      loc-gowalla_totalCheckins.txt
      gowalla_spots_subset1.csv
      us_state_polygon_json.json
      # after running generate_ca_raw.py:
      dataset_gowalla_ca_ne.csv
    /preprocessed
      ... (same as above)
```

- Preprocessing writes to `data/<dataset>/preprocessed/` when you train/test.
- If preprocessed files are missing, they are created automatically before training/evaluation.

## 3) Raw file formats

- **TSMC/Foursquare-style (used by `tky`)**
  - File: TSV with columns (no header):
    - `UserId`, `PoiId`, `PoiCategoryId`, `PoiCategoryName`, `Latitude`, `Longitude`, `TimezoneOffset`, `UTCTime`
  - Time parsing: `%a %b %d %H:%M:%S +0000 %Y` (UTC), then `UTCTimeOffset = UTCTime + TimezoneOffset` (minutes)

- **Gowalla CA+NV CSV (used by `ca`)**
  - File: CSV with header and columns:
    - `UserId`, `PoiId`, `PoiCategoryId`, `Latitude`, `Longitude`, `UTCTime`
  - Time parsing: ISO `YYYY-MM-DDTHH:MM:SSZ` into `UTCTimeOffset` (UTC)

- **NYC pre-split CSVs (used by `nyc`)**
  - Files: `NYC_train.csv`, `NYC_val.csv`, `NYC_test.csv`
  - Columns include (selection):
    - `UserId`, `PoiId`, `PoiCategoryId`, `PoiCategoryCode`, `PoiCategoryName`,
      `Latitude`, `Longitude`, `TimezoneOffset`, `UTCTime`, `UTCTimeOffset`,
      `UTCTimeOffsetWeekday`, `UTCTimeOffsetNormInDayTime`, `pseudo_session_trajectory_id`,
      `UTCTimeOffsetNormDayShift`, `UTCTimeOffsetNormRelativeTime`, `SplitTag`

During preprocessing, additional features are derived, including `UTCTimeOffsetEpoch`, `UTCTimeOffsetHour`, etc., and user/session grouping.

## 4) Preprocessed artifacts

- CSVs under `data/<dataset>/preprocessed/`:
  - `sample.csv`
  - `train_sample.csv`, `validate_sample.csv`, `test_sample.csv`
  - `label_encoding.pkl` (ID encoders and padding IDs)
- PyTorch Geometric data under `data/<dataset>/preprocessed/`:
  - `ci2traj_pyg_data.pt`
  - `traj2traj_pyg_data.pt`

These `.pt` files are PyTorch Geometric Data objects consumed by the loader:
- `ci2traj` provides `x`, `edge_index`, `edge_delta_t`, `edge_delta_s`.
- `traj2traj` provides `x`, `edge_index`, `edge_delta_t`, `edge_delta_s`, `edge_type`, `edge_attr`.

## 5) How to preprocess

- Training/Testing will trigger preprocessing automatically when files are missing.
- Or to prepare the Gowalla CA+NV CSV first:
  ```bash
  python generate_ca_raw.py
  ```

## 6) Adding other datasets

Currently supported dataset names in configs: `nyc`, `tky`, `ca`.

- **Reuse an existing reader**
  - If your data is TSV in the Foursquare/TSMC format, place it at `data/tky/raw/<your_file>.txt` and use `dataset_name: tky`.
  - If your data is CSV matching the Gowalla CA schema, place it at `data/ca/raw/<your_file>.csv` and use `dataset_name: ca`.
  - Put files in `data/<name>/raw` and update thresholds/grouping in `conf/best_conf/<name>.yml` as needed.

- **Add a new dataset name**
  - Extend `preprocess/preprocess_main.py` to handle your `dataset_name` similarly to `nyc`, `tky`, or `ca`.
  - Ensure you output the same preprocessed files (`sample.csv`, split CSVs, `.pt` graphs) under `data/<dataset>/preprocessed/`.

- **Minimum required fields**
  - `UserId`, `PoiId`, `PoiCategoryId`, `Latitude`, `Longitude`, `UTCTime` (UTC), and a time zone offset or a unified UTC time.
  - The pipeline will:
    - derive per-user temporal ordering, split into train/val/test (time-ordered),
    - encode IDs, form (pseudo-)trajectory groups, and
    - build graph inputs (`ci2traj_pyg_data.pt`, `traj2traj_pyg_data.pt`).
