# Bericht – Hinweise und Verknüpfungen

## Word-Bericht (lokal)
Der umfangreiche Word-Entwurf liegt lokal (nicht im Repo), z. B.:
- `.../Projekt NLP/Bericht/` (OneDrive)

## Artefakte aus dem Repo, die im Bericht referenziert werden
### Tabellen
- `reports/tables/eda_counts_category.csv`
- `reports/tables/eda_counts_rating.csv`
- `reports/tables/eda_textlen_summary.csv`
- `reports/tables/val_metrics.csv`
- `reports/tables/test_metrics.csv`
- `reports/tables/error_examples.csv`
- `reports/tables/error_analysis_notes.md`

### Abbildungen
- `reports/figures/confusion_matrix_test.png`
- `reports/figures/confusion_matrix_test_normalized.png`

## Reproduzierbarkeit (Kurz)
1) venv aktivieren, Requirements installieren
2) Pipeline laufen lassen:
- `python src/02_sample_and_eda.py`
- `python src/05_make_features_tfidf.py`
- `python src/06_train_models_val.py`
- `python src/07_eval_test.py`
- `python src/09_confusion_matrix_normalized.py`
- `python src/08_error_analysis_samples.py`

## Word-Hinweis
- Inhaltsverzeichnis: Referenzen → Inhaltsverzeichnis aktualisieren
- Abbildungen: Confusion-Matrix-PNGs aus `reports/figures/` einfügen
