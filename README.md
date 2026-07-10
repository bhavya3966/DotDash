# DotDash

Upload your weekly "Closing Mx List" export and get the Previous Week → New Opportunities →
Subtotal → Reduction → Current Week bridge, team/segment tabs, Revenue by Regional Head
breakdown, and a sortable merchant table. Standalone app — no other features mixed in.

## Folder structure

```
dotdash/
├── main.py
├── requirements.txt
├── render.yaml
├── README.md
└── static/
    └── index.html
```

## How it works

- **Upload the `.xlsx` export directly** — no reformatting needed. It auto-detects the right
  sheet by scanning for weekly revenue columns (`W0`, `W1`, `W_2`, ...) plus a `Regional Head`
  column, and scores candidates so it picks the true master list even when the workbook has
  several similar-looking sheets (per-RH breakdowns, payment mapping exports, etc).
- **New Opportunities / Reduction** are computed per merchant as the week-over-week revenue
  delta, split into gains and losses. Verified against a real screenshot from this workbook —
  matches to the exact lakh. If you'd rather anchor Reduction to a literal `Backup / Reduction`
  column instead of the computed delta, that's a small change in `compute_bridge()` in `main.py`.
- **Two ways to control it**, both driving the same widgets:
  1. Manual — week-transition buttons, team tabs (all teams found in the data, not hardcoded),
     and segment tabs.
  2. Natural language — type something like *"show me Farming team from W0 to W2"* into the
     prompt box; your chosen LLM (Claude / Gemini / GPT, your own key) parses it into the same
     filter parameters and updates the dashboard.
- Team/segment tabs are fully dynamic — whatever's in your sheet's `Team` and `Type of Merchant`
  columns shows up automatically.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000. Manual controls (week buttons, team/segment tabs) work with no
API key at all. The prompt box needs a key — paste one into the settings panel first.

## Deploy on Render

1. Push this folder to its own GitHub repo (separate from any other app).
2. On Render: **New > Web Service** > connect the repo.
3. Render auto-detects `render.yaml` (service name: `dotdash`). If not, set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. No environment variables needed — deploy as-is.
5. First boot on the free tier takes 30-60s (cold start) — open the URL yourself a few minutes
   before you need it so it's warm.

## Known limits

- **In-memory sessions**: uploaded data lives in server RAM, not persisted. A server restart
  clears all sessions.
- **No auth**: anyone with the URL can use it.
- **Merchant table caps at 500 rows** server-side (sorted by current-week revenue), and the UI
  shows the top 50 of those. Fine for review, not for full data export — let me know if you
  want a CSV export button.
- **Client-side key storage**: the prompt box's LLM key lives in the browser's `localStorage`,
  not a secrets vault. Don't reuse a key here that has broader production access if that makes
  you uneasy.
