"""
Metadata loading, splitting and clinical-feature encoding (framework-agnostic).

Clinical features (the "Clinical data" input in the design):
    age          -> median-imputed, standardised
    sex          -> mode-imputed ("unknown" treated as missing), one-hot
    localization -> mode-imputed ("unknown" treated as missing), one-hot
    dx_type      -> one-hot, ONLY in "paper" mode (it leaks the label)

All statistics are fit on TRAIN only and saved to JSON so exactly the same
encoding can be re-applied at inference time (predict.py).
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_and_split_metadata(csv_path=None):
    df = pd.read_csv(csv_path or config.METADATA_CSV)

    if config.SPLIT_MODE == "paper":
        train_df, test_df = train_test_split(
            df, train_size=config.TRAIN_FRAC, stratify=df["dx"],
            random_state=config.RANDOM_SEED)
        print(f"Split (paper 70/30, image-level) -> train {len(train_df)}, test {len(test_df)}")
        return train_df.reset_index(drop=True), None, test_df.reset_index(drop=True)

    # strict: split by lesion so no lesion appears in two splits
    lesions = df.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]]
    train_l, temp_l = train_test_split(
        lesions, train_size=config.TRAIN_FRAC, stratify=lesions["dx"],
        random_state=config.RANDOM_SEED)
    val_share = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_l, test_l = train_test_split(
        temp_l, train_size=val_share, stratify=temp_l["dx"],
        random_state=config.RANDOM_SEED)

    pick = lambda ids: df[df.lesion_id.isin(ids.lesion_id)].reset_index(drop=True)
    train_df, val_df, test_df = pick(train_l), pick(val_l), pick(test_l)
    print(f"Split (strict 70/15/15, lesion-grouped) -> train {len(train_df)}, "
          f"val {len(val_df)}, test {len(test_df)}")
    return train_df, val_df, test_df


def _categorical_cols():
    return ["sex", "localization"] + (["dx_type"] if config.USE_DX_TYPE else [])


def fit_clinical_encoder(train_df):
    """Learn imputation values, category lists and age scaling from TRAIN only."""
    enc = {"age_median": float(train_df["age"].median()), "categories": {}, "modes": {}}
    age = train_df["age"].fillna(enc["age_median"])
    enc["age_mean"] = float(age.mean())
    enc["age_std"] = float(age.std() or 1.0)
    for col in _categorical_cols():
        vals = train_df[col].replace("unknown", np.nan)
        enc["modes"][col] = str(vals.mode()[0])
        enc["categories"][col] = sorted(vals.fillna(enc["modes"][col]).astype(str).unique())
    enc["feature_names"] = ["age"] + [f"{c}_{v}" for c in _categorical_cols()
                                      for v in enc["categories"][c]]
    return enc


def transform_clinical(df, enc):
    """DataFrame rows -> float32 matrix [N, n_features] using a fitted encoder."""
    age = df["age"].fillna(enc["age_median"]).to_numpy(dtype=np.float32)
    parts = [((age - enc["age_mean"]) / enc["age_std"])[:, None]]
    for col, cats in enc["categories"].items():
        vals = df[col].replace("unknown", np.nan).fillna(enc["modes"][col]).astype(str)
        onehot = np.zeros((len(df), len(cats)), dtype=np.float32)
        idx = {c: i for i, c in enumerate(cats)}
        for r, v in enumerate(vals):
            if v in idx:                 # unseen category -> all zeros
                onehot[r, idx[v]] = 1.0
        parts.append(onehot)
    return np.concatenate(parts, axis=1).astype(np.float32)


def save_encoder(enc, path):
    with open(path, "w") as f:
        json.dump(enc, f, indent=2)


def load_encoder(path):
    with open(path) as f:
        return json.load(f)


def labels_of(df):
    idx = {c: i for i, c in enumerate(config.CLASS_NAMES)}
    return df["dx"].map(idx).to_numpy(dtype=np.int32)
