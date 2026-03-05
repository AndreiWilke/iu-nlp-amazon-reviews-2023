# Entscheidungen (DECISIONS)

## Datenbasis
- Dataset: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace Datasets)
- Loader: `trust_remote_code=True` (erforderlich für korrektes Laden des Datasets)

## Reproduzierbarkeit
- Seed: 42
- Splits: stratifiziert nach `rating` (1–5)

## Kategorien
- Electronics
- Home_and_Kitchen
- Clothing_Shoes_and_Jewelry

## Sampling
- Sampling-Strategie: pro Kategorie N Reviews per Streaming aus Split `full`
- N pro Kategorie: 10_000
- Gesamtumfang: 30_000 Reviews

## Textdefinition (Input)
- Textfeld: `text` (Kombination aus Titel + Review-Text; in der Pipeline als ein Feld gespeichert)

## Zielvariable (Label)
- `rating` ∈ {1,2,3,4,5}

## Datensplit (final)
- Train / Validation / Test = 70% / 10% / 20%
- Größen bei N=30_000: Train 21_000, Val 3_000, Test 6_000
- Split-Erzeugung: zweistufig (zuerst 20% Test, dann aus Rest 12.5% als Val), jeweils stratifiziert nach `rating`

## Modellwahl (Kriterium)
- Primäres Kriterium: Validation Macro-F1 (wegen Klassenunwucht, 5★ dominiert)
- Sekundär: Validation Accuracy

## Ergebnis (Kurz, reproduziert)
- Bestes Modell: Logistic Regression mit `class_weight="balanced"`
- Test: Accuracy = 0.7443333, Macro-F1 = 0.5346203