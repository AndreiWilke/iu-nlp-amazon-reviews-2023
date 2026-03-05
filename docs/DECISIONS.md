# Entscheidungen (DECISIONS)

## Datenbasis
- Dataset: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace Datasets)
- trust_remote_code: True (erforderlich für Loader)

## Reproduzierbarkeit
- Seed: 42

## Sampling & Kategorien (Phase 1)
- Geplante Kategorien: Electronics, Home_and_Kitchen, Clothing_Shoes_and_Jewelry
- Ziel-Sample: 10_000 Reviews pro Kategorie (gesamt 30_000), sofern verfügbar
- Split: Train/Test = 80/20, stratifiziert nach rating (1–5)

- Split (aktualisiert): Train/Val/Test = 70/10/20, stratifiziert nach rating (Seed 42)
