import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42

IN_DATA = Path("data/reviews_sample.parquet")

OUT_VEC = Path("models/tfidf_vectorizer.joblib")

OUT_X_TRAIN = Path("data/X_train.npz")
OUT_X_VAL   = Path("data/X_val.npz")
OUT_X_TEST  = Path("data/X_test.npz")

OUT_Y_TRAIN = Path("data/y_train.npy")
OUT_Y_VAL   = Path("data/y_val.npy")
OUT_Y_TEST  = Path("data/y_test.npy")

def dist(y):
    u, c = np.unique(y, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))

def main():
    df = pd.read_parquet(IN_DATA)

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    X_train_text = train_df["text"].astype(str).tolist()
    X_val_text   = val_df["text"].astype(str).tolist()
    X_test_text  = test_df["text"].astype(str).tolist()

    y_train = train_df["rating"].astype(int).to_numpy()
    y_val   = val_df["rating"].astype(int).to_numpy()
    y_test  = test_df["rating"].astype(int).to_numpy()

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=100_000,
        sublinear_tf=True,
    )

    X_train = vec.fit_transform(X_train_text)   # nur train fitten
    X_val   = vec.transform(X_val_text)
    X_test  = vec.transform(X_test_text)

    os.makedirs("models", exist_ok=True)
    joblib.dump(vec, OUT_VEC)

    sparse.save_npz(OUT_X_TRAIN, X_train)
    sparse.save_npz(OUT_X_VAL,   X_val)
    sparse.save_npz(OUT_X_TEST,  X_test)

    np.save(OUT_Y_TRAIN, y_train)
    np.save(OUT_Y_VAL,   y_val)
    np.save(OUT_Y_TEST,  y_test)

    print("[OK] Gespeichert:")
    print(" -", OUT_VEC)
    print(" -", OUT_X_TRAIN, X_train.shape, "y:", y_train.shape, "dist:", dist(y_train))
    print(" -", OUT_X_VAL,   X_val.shape,   "y:", y_val.shape,   "dist:", dist(y_val))
    print(" -", OUT_X_TEST,  X_test.shape,  "y:", y_test.shape,  "dist:", dist(y_test))

if __name__ == "__main__":
    main()
