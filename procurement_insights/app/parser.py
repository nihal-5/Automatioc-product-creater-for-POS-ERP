from __future__ import annotations

import io
import re
from typing import List, Tuple
from pdfminer.high_level import extract_text
from .models import InvoiceParse, LineItem, Totals


CURRENCY_RE = re.compile(r"\b(?:USD|EUR|GBP|INR|Rs\.?|₹|\$|€|£)\b", re.I)
INVOICE_NO_RE = re.compile(r"(invoice\s*(no\.?|number)\s*[:#]?\s*)([A-Z0-9\-_/]+)", re.I)
DATE_RE = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
GST_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b")
VAT_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{8,12}\b")
BARCODE_RE = re.compile(r"\b\d{12,14}\b")


def _extract_text(content: bytes) -> str:
    with io.BytesIO(content) as f:
        return extract_text(f) or ""


def extract_text_from_pdf(content: bytes) -> str:
    """Public helper to extract text from PDF bytes."""
    return _extract_text(content)


def _parse_currency(text: str) -> str | None:
    m = CURRENCY_RE.search(text)
    if not m:
        return None
    raw = m.group(0)
    if raw.lower().startswith("rs") or "₹" in raw:
        return "INR"
    if raw.startswith("$"):
        return "USD"
    if raw == "€":
        return "EUR"
    if raw == "£":
        return "GBP"
    return raw.upper()


def _parse_invoice_id(text: str) -> str | None:
    m = INVOICE_NO_RE.search(text)
    return m.group(3).strip() if m else None


def _parse_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    return m.group(1) if m else None


def _parse_tax_id(text: str) -> str | None:
    m = GST_RE.search(text)
    if m:
        return m.group(0)
    m = VAT_RE.search(text)
    return m.group(0) if m else None


def _parse_totals(text: str) -> Totals:
    subtotal = _grab_amount(text, r"sub\s*total[:\s]*([\d,]+\.\d{2}|\d+)")
    tax = _grab_amount(text, r"(tax|gst|vat)[:\s]*([\d,]+\.\d{2}|\d+)")
    total = _grab_amount(text, r"(grand\s*total|total\s*amount|total)[:\s]*([\d,]+\.\d{2}|\d+)")
    currency = _parse_currency(text)
    return Totals(
        subtotal=subtotal,
        tax=tax if tax is not None else None,
        total=total,
        currency=currency,
    )


def _grab_amount(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, re.I)
    if not m:
        return None
    amt_str = m.group(m.lastindex or 1)
    amt_str = amt_str.replace(",", "").strip()
    try:
        return float(amt_str)
    except ValueError:
        return None


def _parse_line_items(text: str) -> List[LineItem]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: List[LineItem] = []
    line_re = re.compile(
        r"(?P<desc>[A-Za-z0-9][A-Za-z0-9 ,.\-_/]{2,})\s+"
        r"(?P<qty>\d+(?:\.\d+)?)\s+"
        r"(?P<unit_price>\d+(?:\.\d+)?)\s+"
        r"(?P<line_total>\d+(?:\.\d+)?)"
    )
    for ln in lines:
        m = line_re.search(ln)
        if not m:
            continue
        try:
            qty = float(m.group("qty"))
            unit_price = float(m.group("unit_price"))
            line_total = float(m.group("line_total"))
        except ValueError:
            continue
        desc = m.group("desc").strip(" -")
        items.append(
            LineItem(
                description=desc,
                quantity=qty,
                unit_price=unit_price,
                line_total=line_total,
            )
        )
    return items


def _calc_anomalies(line_items: List[LineItem], totals: Totals, tax_id: str | None) -> List[str]:
    anomalies: List[str] = []
    if not tax_id:
        anomalies.append("Supplier tax ID missing.")
    if not totals.total:
        anomalies.append("Total amount missing.")
    if totals.total is not None and totals.total <= 0:
        anomalies.append("Total is zero or negative.")
    line_sum = sum(li.line_total or 0 for li in line_items)
        if totals.total and line_sum:
            diff = abs(totals.total - line_sum)
            if diff > max(1.0, line_sum * 0.02):
                anomalies.append(f"Total mismatch vs line sum (diff={diff:.2f}).")
        if totals.tax and totals.subtotal:
            rate = totals.tax / totals.subtotal if totals.subtotal else 0
            if rate < 0 or rate > 0.4:
                anomalies.append(f"Suspicious tax rate ({rate:.2%}).")
        if not line_items:
            anomalies.append("No line items detected.")
        return anomalies


def parse_invoice_text(text: str, file_name: str) -> InvoiceParse:
    """Parse already-extracted text into structured invoice data."""
    invoice_id = _parse_invoice_id(text)
    invoice_date = _parse_date(text)
    tax_id = _parse_tax_id(text)
    totals = _parse_totals(text)
    line_items = _parse_line_items(text)
    barcodes = list({m.group(0) for m in BARCODE_RE.finditer(text)})
    anomalies = _calc_anomalies(line_items, totals, tax_id)

    preview = "\n".join(text.splitlines()[:15])

    parsed = InvoiceParse(
        file_name=file_name,
        text_preview=preview,
        invoice_number=invoice_id,
        invoice_date=invoice_date,
        supplier_tax_id=tax_id,
        barcodes=barcodes,
        line_items=line_items,
        totals=totals,
        anomalies=anomalies,
    )
    return parsed


def process_pdf_bytes(content: bytes, file_name: str) -> dict:
    """Legacy helper: extract text then parse."""
    text = extract_text_from_pdf(content)
    parsed = parse_invoice_text(text, file_name)
    return parsed.model_dump()
