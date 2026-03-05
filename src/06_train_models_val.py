import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

SEED = 42

X_TRAIN = sparse.load_npz("data/X_train.npz")
X_VAL   = sparse.load_npz("data/X_val.npz")

y_train = np.load("data/y_train.npy")
y_val   = np.load("data/y_val.npy")

Path("models").mkdir(exist_ok=True)
Path("reports/tables").mkdir(parents=True, exist_ok=True)

def eval_model(model):
    model.fit(X_TRAIN, y_train)
    pred = model.predict(X_VAL)
    acc = accuracy_score(y_val, pred)
    f1m = f1_score(y_val, pred, average="macro")
    return acc, f1m

def save(model, path):
    joblib.dump(model, path)

def main():
    results = []

    # --- Logistic Regression (multiclass) ---
    logreg_normal = LogisticRegression(
        max_iter=2000,
        C=4.0,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight=None,
        random_state=SEED,
    )
    acc, f1m = eval_model(logreg_normal)
    path = "models/logreg_normal.joblib"
    save(logreg_normal, path)
    results.append({"model": "logreg_normal", "val_accuracy": acc, "val_macro_f1": f1m, "path": path})

    logreg_balanced = LogisticRegression(
        max_iter=2000,
        C=4.0,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        random_state=SEED,
    )
    acc, f1m = eval_model(logreg_balanced)
    path = "models/logreg_balanced.joblib"
    save(logreg_balanced, path)
    results.append({"model": "logreg_balanced", "val_accuracy": acc, "val_macro_f1": f1m, "path": path})

    # --- Linear SVM ---
    lsvm_normal = LinearSVC(C=1.0, class_weight=None, random_state=SEED)
    acc, f1m = eval_model(lsvm_normal)
    path = "models/linearsvm_normal.joblib"
    save(lsvm_normal, path)
    results.append({"model": "linearsvm_normal", "val_accuracy": acc, "val_macro_f1": f1m, "path": path})

    lsvm_balanced = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
    acc, f1m = eval_model(lsvm_balanced)
    path = "models/linearsvm_balanced.joblib"
    save(lsvm_balanced, path)
    results.append({"model": "linearsvm_balanced", "val_accuracy": acc, "val_macro_f1": f1m, "path": path})

    df = pd.DataFrame(results).sort_values(["val_macro_f1", "val_accuracy"], ascending=False)
    df.to_csv("reports/tables/val_metrics.csv", index=False)

    best_name = df.iloc[0]["model"]
    best_path = df.iloc[0]["path"]
    best_model = joblib.load(best_path)
    joblib.dump(best_model, "models/best_model.joblib")

    with open("models/model_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(df[["model", "val_accuracy", "val_macro_f1"]].to_string(index=False))
    print("\n[OK] Best model:", best_name)
    print("[OK] Saved as models/best_model.joblib from:", best_path)

if __name__ == "__main__":
    main()
