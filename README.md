\# IU NLP Projekt – Amazon Reviews ’23 (1–5 Sterne)



Dieses Repository enthält eine reproduzierbare Pipeline zur Sentimentanalyse / 1–5-Sterne-Klassifikation auf Amazon Reviews ’23 (McAuley Lab).  

Kategorien: \*\*Electronics\*\*, \*\*Home\_and\_Kitchen\*\*, \*\*Clothing\_Shoes\_and\_Jewelry\*\*.  

Split: \*\*Train/Val/Test = 70/10/20\*\* (stratifiziert nach Rating), \*\*Seed = 42\*\*.



\## Projektstruktur

\- `src/` – Skripte für Sampling, Features, Training, Evaluation

\- `data/` – \*\*nicht versioniert\*\* (wird erzeugt)

\- `models/` – \*\*nicht versioniert\*\* (wird erzeugt)

\- `reports/tables/` – CSV-Tabellen (EDA, Metriken, Fehlerbeispiele)

\- `reports/figures/` – PNG-Abbildungen (Confusion Matrices)

\- `docs/DECISIONS.md` – getroffene Entscheidungen (Seed, Split, Kategorien)

\- `requirements.txt` – Python-Abhängigkeiten



\## Setup (Windows / PowerShell)

```powershell

py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

