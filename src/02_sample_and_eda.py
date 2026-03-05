import os
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

DATASET = "McAuley-Lab/Amazon-Reviews-2023"
CONFIGS = {
    "Electronics": "raw_review_Electronics",
    "Home_and_Kitchen": "raw_review_Home_and_Kitchen",
    "Clothing_Shoes_and_Jewelry": "raw_review_Clothing_Shoes_and_Jewelry",
}
N_PER_CAT = 10_000

OUT_DATA = Path("data/reviews_sample.parquet")
OUT_RATING = Path("reports/tables/eda_counts_rating.csv")
OUT_CAT = Path("reports/tables/eda_counts_category.csv")
OUT_LEN = Path("reports/tables/eda_textlen_summary.csv")

def get_streaming_split(cfg: str):
    ds_dict = load_dataset(DATASET, cfg, streaming=True, trust_remote_code=True)
    split_names = list(ds_dict.keys())
    if not split_names:
        raise RuntimeError(f"Keine Splits gefunden für Config {cfg}")
    chosen = split_names[0]
    return chosen, ds_dict[chosen]

def stream_sample(stream_ds, n: int) -> list[dict]:
    rows = []
    for ex in stream_ds:
        rating = ex.get("rating") or ex.get("stars") or ex.get("overall")
        title = ex.get("title") or ""
        text = ex.get("text") or ex.get("reviewText") or ex.get("review_text") or ""
        full_text = (str(title).strip() + " " + str(text).strip()).strip()

        if rating is None or full_text == "":
            continue

        try:
            rating_int = int(rating)
        except Exception:
            continue

        if not (1 <= rating_int <= 5):
            continue

        rows.append({"rating": rating_int, "text": full_text})
        if len(rows) >= n:
            break
    return rows

def main():
    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    all_rows = []

    for cat, cfg in CONFIGS.items():
        split_name, stream_ds = get_streaming_split(cfg)
        print(f"[INFO] {cat} -> {cfg} | split={split_name}")

        rows = stream_sample(stream_ds, N_PER_CAT)
        print(f"[INFO] Gesampelt: {len(rows)}")

        for r in rows:
            r["category"] = cat
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("Sample ist leer. Prüfe Feldnamen/Configs.")

    # 70/10/20 via 2-step split:
    # 1) Test 20%
    temp_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["rating"]
    )
    # 2) Val 10% gesamt => 0.10/0.80 = 0.125 vom Rest
    train_df, val_df = train_test_split(
        temp_df, test_size=0.125, random_state=SEED, stratify=temp_df["rating"]
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    df2 = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df2.to_parquet(OUT_DATA, index=False)

    rating_counts = (
        df2.groupby(["split", "rating"])
        .size()
        .reset_index(name="n")
        .sort_values(["split", "rating"])
    )
    rating_counts.to_csv(OUT_RATING, index=False)

    cat_counts = (
        df2.groupby(["split", "category"])
        .size()
        .reset_index(name="n")
        .sort_values(["split", "category"])
    )
    cat_counts.to_csv(OUT_CAT, index=False)

    df2["text_len"] = df2["text"].astype(str).str.len()
    len_summary = (
        df2.groupby("split")["text_len"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
    )
    len_summary.to_csv(OUT_LEN, index=False)

    print("\n[OK] Dateien geschrieben:")
    print(" -", OUT_DATA)
    print(" -", OUT_RATING)
    print(" -", OUT_CAT)
    print(" -", OUT_LEN)
    print("\n[INFO] Split sizes:", df2["split"].value_counts().to_dict())

if __name__ == "__main__":
    main()
