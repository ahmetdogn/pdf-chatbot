# pdf-chatbot

Ein kleines Projekt, mit dem man PDFs hochladen und Fragen dazu stellen kann. Die Antworten kommen von Claude, und hinter jeder Aussage steht die Quelle mit Seitenzahl. Wenn man auf die Quelle klickt, sieht man die Textstelle aus dem PDF.

**Live Demo:** https://pdf-chatbot-ashen-pi.vercel.app

![Screenshot](docs/screenshot.png)

## Wie es funktioniert

1. Das PDF wird an `/api/parse` geschickt, der Server liest den Text Seite für Seite aus (pypdf)
2. Der Text wird in Chunks von ca. 900 Zeichen aufgeteilt. Ein Chunk geht nie über eine Seite hinaus, damit die Seitenzahl beim Zitieren stimmt
3. Im Browser werden mit [transformers.js](https://github.com/huggingface/transformers.js) die Embeddings berechnet (mehrsprachiges MiniLM-Modell, kostenlos, kein extra API Key) und alles in IndexedDB gespeichert
4. Bei einer Frage schickt der Browser Frage + Chunks an `/api/chat`, der Server sucht dann mit:
   - BM25 (normale Stichwortsuche)
   - Vektorsuche (findet auch Sachen, die anders formuliert sind)
5. Beide Ergebnislisten werden mit Reciprocal Rank Fusion zusammengeführt
6. Die besten 6 Chunks + Frage gehen an Claude, die Antwort wird gestreamt

Der Server speichert also gar nichts. Am Anfang hatte ich ChromaDB + fastembed auf dem Server, aber das lief auf Vercel nicht, weil Serverless-Funktionen keinen festen Speicher haben und das Modell zu groß war. Deshalb laufen die Embeddings jetzt im Browser. Falls das Modell nicht laden kann, geht die App trotzdem, dann halt nur mit BM25.

Problem, auf das ich gestoßen bin: Die normale BM25-Formel (`BM25Okapi`) gibt Wörtern, die in mehr als der Hälfte der Chunks vorkommen, eine IDF von 0 oder sogar negativ. Bei kleinen PDFs mit wenigen Chunks sind dadurch manchmal genau die richtigen Treffer rausgefallen. Ich hab deshalb die IDF-Formel von Lucene genommen, die immer positiv ist (`app/retrieval.py`).

## Projektstruktur

```
app/
  main.py        FastAPI Routen (Upload, Dokumente, Chat)
  pdf_loader.py  Text aus PDF holen
  chunking.py    Text in Chunks aufteilen
  embeddings.py  fastembed + Hash-Embedder (nur für eval und tests)
  retrieval.py   BM25, Vektorsuche, Fusion
  llm.py         Claude API
public/          Frontend (HTML/CSS/JS, kein Framework)
eval/            Testfragen + Evaluation
tests/
```

## Starten

Man braucht Python 3.11+ und einen API Key von https://console.anthropic.com

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # API Key eintragen
uvicorn app.main:app --reload
```

Dann http://localhost:8000 öffnen. Beim ersten Upload lädt der Browser das Embedding-Modell (ca. 120 MB), danach ist es gecached.

Oder mit Docker:

```bash
docker compose up --build
```

## Auf Vercel deployen

1. Repo auf vercel.com importieren (Vercel erkennt FastAPI automatisch)
2. Unter Settings > Environment Variables `ANTHROPIC_API_KEY` eintragen
3. Am besten auch `DEMO_PASSWORD` setzen, sonst kann jeder mit deinem API Key chatten
4. Deploy

PDFs dürfen auf Vercel max. 4 MB groß sein (Limit für Requests ist 4.5 MB).

## Evaluation

Ich wollte wissen, ob die Hybrid-Suche wirklich besser ist als nur eine der beiden Suchen. Dafür gibt es in `eval/` ein ausgedachtes Mitarbeiterhandbuch (6 Seiten) und 20 Fragen, bei denen jeweils die richtige Seite hinterlegt ist. Die Fragen sind extra anders formuliert als der Text, z.B. "Phishing-Mail" statt "verdächtige E-Mails".

```bash
pip install -e ".[eval]"
python -m eval.run_eval
```

Die Evaluation benutzt fastembed mit dem gleichen Modell wie der Browser. Ergebnis mit `--embedder hash` (einfacher Embedder ohne echtes Modell):

| Modus  | Hit@3 | MRR  |
|--------|-------|------|
| BM25   | 75%   | 0.72 |
| Vektor | 90%   | 0.82 |
| Hybrid | 95%   | 0.87 |

Hit@3 = richtige Seite ist unter den ersten 3 Treffern. MRR = wie weit oben der richtige Treffer im Schnitt ist (1.0 = immer ganz oben).

TODO: Ergebnis mit dem echten fastembed Modell eintragen

## Tests

```bash
pytest
ruff check .
```

Die Tests brauchen keinen API Key, statt Claude wird ein Fake benutzt und statt fastembed der Hash-Embedder.

## Einstellungen (.env)

| Variable | Standard |
|---|---|
| `ANTHROPIC_API_KEY` | |
| `DEMO_PASSWORD` | leer (= kein Passwort) |
| `CLAUDE_MODEL` | `claude-opus-5-5` |
| `CLAUDE_EFFORT` | `low` |
| `TOP_K` | `6` |
| `CHUNK_SIZE` | `900` |
| `CHUNK_OVERLAP` | `150` |

## Ideen

- OCR für gescannte PDFs
- Reranking mit einem Cross-Encoder
- Login, damit jeder nur seine eigenen PDFs sieht
