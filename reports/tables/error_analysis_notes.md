# Fehleranalyse (Test) – qualitative Beispiele

## Beobachtung 1: Grenzfälle 3↔4 und 4↔5
Viele Reviews enthalten gleichzeitig positive und negative Hinweise (z. B. "Nice but...", "looks great, but...").
Das Modell reagiert stark auf sentiment-tragende Wörter und verschiebt Grenzfälle häufig um 1 Stern.

## Beobachtung 2: Einzelne Negativtrigger in sonst positiven Reviews
Wörter wie "unfortunately", "odor", "glue", "thin" können die Vorhersage nach unten drücken, obwohl Nutzer dennoch 5 Sterne vergeben.

## Beobachtung 3: Extreme Fehler durch inkonsistente Texte / Updates
Einige Reviews enthalten Updates oder widersprüchliche Abschnitte (z. B. erst positives Fazit, später "no longer working").
TF-IDF kann die zeitliche Struktur nicht modellieren, dadurch entstehen seltene extreme Fehlklassifikationen.

## Beispiele
Siehe: reports/tables/error_examples.csv
