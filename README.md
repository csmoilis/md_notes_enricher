# MD Notes Enricher

Turn study notes written in Markdown into an **enriched** document (definitions, explanations,
context) using **Gemini Flash** on the free API tier.

## How it works

1. **Global context** – the whole `.md` file is sent once to Gemini, which returns a summary,
   the key points and how the concepts relate. This is passed as the `system_instruction`
   of every later call, so each chunk is enriched with the whole document in mind.
2. **Semantic chunking** – the notes are split at Markdown headers (`#`, `##`, `###`), never in
   the middle of a paragraph or code block. Small sections are merged and big ones are split at
   deeper headers (`####`…) or paragraph breaks, giving chunks of ~1,000–2,000 words.
   `#` lines inside code blocks are ignored; text before the first header is kept.
3. **Iterative prompting** – each chunk is sent with:
   *"Using the global context provided, enrich the following specific subsection with
   definitions and relevant subject information: [CHUNK]"*. The model keeps your original
   content and headings and adds to them.
4. **Rate limits** – a 5-second pause between requests (under 12 requests/minute) plus
   exponential backoff with jitter on HTTP 429 / `RESOURCE_EXHAUSTED` and 503 errors, honouring
   the retry delay Google sends. If the **daily** free quota is used up it stops cleanly.
5. **Reassembly** – each enriched chunk is appended to the output file as soon as it is ready,
   so nothing is lost if it stops. Use `--resume` (or the **Resume** button) to continue.

## Setup (Windows)

```bat
cd md_notes_enricher
python -m venv env
env\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
notepad .env
```

On macOS/Linux use `python3 -m venv env` and `env/bin/python -m pip install -r requirements.txt`.

### Get a free Gemini API key

1. Go to <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Paste it into `.env` as `GEMINI_API_KEY=...` (or set it as an environment variable).
   `.env` is git-ignored — never commit your key.

> Free-tier note: prompts sent on the free tier may be used by Google to improve its products,
> and the free tier has per-minute **and per-day** request limits per model. Each run uses
> *number of chunks + 1* requests (`--dry-run` shows the count). If you hit the daily cap,
> run again the next day with `--resume`, or use a model with a bigger free quota such as
> `gemini-3.5-flash-lite`. Check current limits at <https://ai.google.dev/gemini-api/docs/rate-limits>.

## Use it

### App (Streamlit)

Double-click **`run_app.bat`** (Windows) or run `./run_app.sh` (Git Bash / macOS / Linux).
Then in the browser: upload your `.md`, check the chunk plan, click **Enrich notes**, watch
the progress bar, preview and **Download enriched .md**. Results are also saved in `outputs/`.

### Command line

```bat
env\Scripts\python.exe enrich.py notes.md -o output.md
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-o, --output` | `<input>_enriched.md` | output file |
| `-m, --model` | `gemini-3.8-flash` | any Gemini model id |
| `-d, --delay` | `5` | seconds between requests |
| `--max-words` / `--min-words` | `2000` / `1000` | chunk size range |
| `-l, --language` | same as notes | language for the additions, e.g. `English` |
| `--include-summary` | off | put the global summary at the top of the output |
| `--resume` | off | continue an interrupted run |
| `--retries` | `6` | max retries per request on 429/503 |
| `--dry-run` | off | only show the chunks, no API calls |

## Tests

```bat
env\Scripts\python.exe -m pytest
```

The tests mock the Gemini client (no key needed). If `GEMINI_API_KEY` is set, one tiny live test
also runs.

## Files

- `chunker.py` – header-based semantic chunking
- `enricher.py` – pipeline, prompts, rate limiting / backoff, resume
- `enrich.py` – command line
- `app.py` – Streamlit UI
- `run_app.bat` / `run_app.sh` – launchers using the `env` virtual environment
