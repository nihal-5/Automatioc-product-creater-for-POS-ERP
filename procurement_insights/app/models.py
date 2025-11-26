from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(..., description="Item description")
    quantity: Optional[float] = Field(None, description="Quantity")
    unit_price: Optional[float] = Field(None, description="Unit price")
    line_total: Optional[float] = Field(None, description="Line total")


class Totals(BaseModel):
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None


class InvoiceParse(BaseModel):
    file_name: str
    text_preview: str
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    supplier_tax_id: Optional[str] = None
    customer_tax_id: Optional[str] = None
    barcodes: List[str] = []
    line_items: List[LineItem] = []
    totals: Totals = Totals()
    anomalies: List[str] = []
