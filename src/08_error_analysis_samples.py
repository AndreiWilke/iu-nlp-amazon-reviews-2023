from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

MODEL_PATH = "models/best_model.joblib"

X_TEST = sparse.load_npz("data/X_test.npz")
y_test = np.load("data/y_test.npy")

Path("reports/tables").mkdir(parents=True, exist_ok=True)

def main():
    # Wir brauchen die Originaltexte wieder:
    df = pd.read_parquet("data/reviews_sample.parquet")
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_TEST)

    test_df["y_true"] = y_test
    test_df["y_pred"] = pred
    test_df["correct"] = (test_df["y_true"] == test_df["y_pred"])

    errors = test_df[~test_df["correct"]].copy()

    def pick(a, b, n=3):
        sub = errors[(errors["y_true"] == a) & (errors["y_pred"] == b)]
        return sub.head(n)

    samples = pd.concat([
        pick(3, 4, 3),
        pick(4, 3, 3),
        pick(4, 5, 3),
        pick(5, 4, 3),
        pick(1, 5, 3),
        pick(5, 1, 3),
    ], ignore_index=True)

    # Kürzen fürs Lesen im Bericht
    samples["text_snippet"] = samples["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 280)

    out = samples[["category", "y_true", "y_pred", "text_snippet"]]
    out.to_csv("reports/tables/error_examples.csv", index=False)

    print("[OK] error examples saved: reports/tables/error_examples.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
