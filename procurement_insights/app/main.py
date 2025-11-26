from __future__ import annotations

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .pipeline import analyze_document

app = FastAPI(title="Procurement Insights", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Procurement Insights</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #1f2937; }
    h1 { margin-bottom: 6px; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    .result { white-space: pre; font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; background: #f9fafb; padding: 14px; border-radius: 10px; overflow: auto; }
    button { background: #2563eb; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    input[type=file] { margin: 12px 0; }
  </style>
</head>
<body>
  <h1>Procurement Insights</h1>
  <p>Upload a supplier invoice (PDF) to extract line items, totals, tax IDs, and anomalies.</p>
  <div class="card">
    <input id="file" type="file" accept="application/pdf">
    <button onclick="upload()">Process</button>
    <div id="msg" style="margin-top:12px;color:#dc2626;"></div>
    <h3>Result</h3>
    <div id="result" class="result">Awaiting upload...</div>
  </div>
  <script>
    async function upload() {
      const fileInput = document.getElementById('file');
      const msg = document.getElementById('msg');
      const result = document.getElementById('result');
      msg.textContent = '';
      if (!fileInput.files.length) {
        msg.textContent = 'Select a PDF first.';
        return;
      }
      const data = new FormData();
      data.append('file', fileInput.files[0]);
      result.textContent = 'Processing...';
      try {
        const res = await fetch('/api/process', { method: 'POST', body: data });
        if (!res.ok) throw new Error('Request failed');
        const json = await res.json();
        result.textContent = JSON.stringify(json, null, 2);
      } catch (e) {
        msg.textContent = 'Upload failed: ' + e.message;
        result.textContent = 'Awaiting upload...';
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    try:
        analysis = analyze_document(content, file.filename)
        return JSONResponse(analysis.model_dump())
    except Exception as exc:  # pragma: no cover - surface as 500
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}")


@app.post("/api/validate")
async def validate(payload: dict):
    # LLM validation is optional; disabled unless OPENAI_API_KEY is set and openai package is available.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"status": "disabled", "note": "Set OPENAI_API_KEY to enable LLM validation."}
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return {"status": "disabled", "note": "openai package not installed; pip install openai to enable."}

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are auditing an invoice JSON for consistency. "
        "Identify any mismatches between totals, taxes, and line items. "
        "Return a short list of issues and a brief explanation. "
        "If everything looks fine, say so."
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(payload)},
            ],
        )
        content = resp.choices[0].message.content if resp.choices else ""
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "note": f"LLM call failed: {exc}"}

    return {"status": "ok", "llm_feedback": content}
