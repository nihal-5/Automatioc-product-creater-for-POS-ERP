from __future__ import annotations

from typing import List
from .models import AnalysisResult, DocumentClassification, ValidationIssue
from . import parser


def classify_text(text: str) -> DocumentClassification:
    lowered = text.lower()
    score = 0.0
    notes: List[str] = []
    if "invoice" in lowered or "bill" in lowered:
        score += 0.55
        notes.append("Found keyword 'invoice' or 'bill'.")
    if "tax" in lowered or "gst" in lowered or "vat" in lowered:
        score += 0.15
        notes.append("Found tax terminology.")
    if "total" in lowered and "amount" in lowered:
        score += 0.1
    if "qty" in lowered or "quantity" in lowered:
        score += 0.1
    doc_type = "invoice" if score >= 0.5 else "unknown"
    return DocumentClassification(doc_type=doc_type, confidence=round(min(1.0, score), 2), notes=notes)


def validate_invoice(anomalies: list[str], classification: DocumentClassification) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if classification.doc_type != "invoice":
        issues.append(ValidationIssue(code="not_invoice", message="Document did not classify as invoice.", severity="error"))
    for a in anomalies:
        issues.append(ValidationIssue(code="anomaly", message=a, severity="warning"))
    return issues


def analyze_document(content: bytes, file_name: str) -> AnalysisResult:
    text = parser.extract_text_from_pdf(content)
    classification = classify_text(text)
    invoice = parser.parse_invoice_text(text, file_name)
    issues = validate_invoice(invoice.anomalies, classification)
    result = AnalysisResult(
        file_name=file_name,
        classification=classification,
        invoice=invoice,
        issues=issues,
    )
    return result
