\# Entscheidungen (DECISIONS)



\## Datenbasis

\- Dataset: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace Datasets)

\- Loader: `trust\_remote\_code=True` (erforderlich für korrektes Laden des Datasets)



\## Reproduzierbarkeit

\- Seed: 42

\- Splits: stratifiziert nach `rating` (1–5)



\## Kategorien

\- Electronics

\- Home\_and\_Kitchen

\- Clothing\_Shoes\_and\_Jewelry



\## Sampling

\- Sampling-Strategie: pro Kategorie N Reviews per Streaming aus Split `full`

\- N pro Kategorie: 10\_000

\- Gesamtumfang: 30\_000 Reviews



\## Textdefinition (Input)

\- Textfeld: `text` (Kombination aus Titel + Review-Text; in der Pipeline als ein Feld gespeichert)



\## Zielvariable (Label)

\- `rating` ∈ {1,2,3,4,5}



\## Datensplit (final)

\- Train / Validation / Test = 70% / 10% / 20%

\- Größen bei N=30\_000: Train 21\_000, Val 3\_000, Test 6\_000

\- Split-Erzeugung: zweistufig (zuerst 20% Test, dann aus Rest 12.5% als Val), jeweils stratifiziert nach `rating`



\## Modellwahl (Kriterium)

\- Primäres Kriterium: Validation Macro-F1 (wegen Klassenunwucht, 5★ dominiert)

\- Sekundär: Validation Accuracy



\## Ergebnis (Kurz, reproduziert)

\- Bestes Modell: Logistic Regression mit `class\_weight="balanced"`

\- Test: Accuracy = 0.7443333, Macro-F1 = 0.5346203

