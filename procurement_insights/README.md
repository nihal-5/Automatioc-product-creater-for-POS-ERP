# Procurement Insights (Invoice/Bill AI)

Upload supplier invoices (PDF) and get structured line items, totals, tax/GST IDs, and anomaly flags. Architecture is agent-like: classify → parse → validate. Runs deterministic parsing with pdfminer + regex heuristics; LLM is optional for validation/explanations (env toggle).

## Features
- Agent-style pipeline: classify document → parse invoice → validate/anomalies.
- PDF text extraction via pdfminer.six.
- Heuristic parsing: invoice number/date, supplier GST/VAT, line items (qty/unit/unit_price/line_total), subtotal/tax/total.
- Anomaly flags: missing tax ID, total mismatch vs. line sums, missing totals, zero/negative totals, suspicious tax rate.
- Barcode detection from raw text (EAN/UPC-like patterns).
- FastAPI endpoints:
  - `GET /` simple UI (HTMX) for upload and results.
  - `POST /api/process` (multipart PDF) → classification + structured JSON + anomalies.
  - `POST /api/validate` (optional) → LLM-based sanity check (disabled by default).
- Lightweight UI (no frontend build).

## Quick start
```bash
cd procurement_insights
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Open http://localhost:8001 and upload a PDF invoice.

API (curl):
```bash
curl -F "file=@/path/to/invoice.pdf" http://localhost:8001/api/process
```

## Config (env)
- `OPENAI_API_KEY` (optional) — set to enable LLM validation endpoint; if unset, `/api/validate` returns a noop response.

## Project layout
```
procurement_insights/
  app/
    main.py      # FastAPI app, UI + process/validate endpoints
    models.py    # Pydantic schemas
    parser.py    # pdfminer extraction + regex/heuristics + anomaly checks
    pipeline.py  # classify -> parse -> validate orchestration
  docs/
    diagram.txt  # brief pipeline outline
  samples/
    sample_response.json
    sample_invoice.pdf (add your own or generate)
  requirements.txt
  README.md
```

## Topics (set in GitHub UI)
`pdf`, `pdfminer`, `fastapi`, `procurement`, `invoice-processing`, `ocr-free`, `regex`, `validation`, `llm-optional`

## Notes
- The parser is heuristic: it favors robustness over perfection and avoids LLM calls by default.
- For LLM validation, add your key and hit `/api/validate` with the parsed JSON to get a short “issues” list and rationale.
