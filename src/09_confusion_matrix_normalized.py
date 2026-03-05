from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODEL_PATH = "models/best_model.joblib"

X_TEST = sparse.load_npz("data/X_test.npz")
y_test = np.load("data/y_test.npy")

Path("reports/figures").mkdir(parents=True, exist_ok=True)

def main():
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_TEST)

    labels = [1,2,3,4,5]
    cm = confusion_matrix(y_test, pred, labels=labels, normalize="true")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format=".2f", colorbar=False)
    ax.set_title("Confusion Matrix normalized (Test) - row-wise")
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix_test_normalized.png", dpi=200)
    plt.close(fig)

    print("[OK] Saved: reports/figures/confusion_matrix_test_normalized.png")

if __name__ == "__main__":
    main()
