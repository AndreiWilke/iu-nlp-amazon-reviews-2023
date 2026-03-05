from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODEL_PATH = "models/best_model.joblib"

X_TEST = sparse.load_npz("data/X_test.npz")
y_test = np.load("data/y_test.npy")

Path("reports/tables").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

def main():
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_TEST)

    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")

    # Save metrics table
    df = pd.DataFrame([{
        "model": "best_model",
        "model_path": MODEL_PATH,
        "test_accuracy": acc,
        "test_macro_f1": f1m
    }])
    df.to_csv("reports/tables/test_metrics.csv", index=False)

    # Confusion matrix
    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_test, pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (Test) - 1 to 5 Stars")
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix_test.png", dpi=200)
    plt.close(fig)

    print("[OK] Test metrics:")
    print(df.to_string(index=False))
    print("[OK] Confusion matrix saved: reports/figures/confusion_matrix_test.png")

if __name__ == "__main__":
    main()
