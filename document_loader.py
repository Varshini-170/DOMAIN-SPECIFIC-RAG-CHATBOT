"""PDF validation and page-level text extraction."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Iterable

from pypdf import PdfReader


class PDFValidationError(ValueError):
    """Raised when an upload is not a usable PDF for this application."""


@dataclass(frozen=True)
class PageDocument:
    """One non-empty source page with citation metadata."""

    text: str
    metadata: dict[str, Any]


def _clean_text(text: str) -> str:
    """Normalise whitespace while retaining paragraph boundaries."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_upload_bytes(upload: Any) -> bytes:
    """Support Streamlit UploadedFile and simple test doubles."""
    if hasattr(upload, "getvalue"):
        return upload.getvalue()
    if hasattr(upload, "read"):
        return upload.read()
    raise PDFValidationError("An uploaded file could not be read.")


def extract_pages(uploaded_files: Iterable[Any], max_file_size_mb: int = 10) -> list[PageDocument]:
    """Extract text from PDFs and preserve document name, page, and file hash."""
    files = list(uploaded_files)
    if not files:
        raise PDFValidationError("Upload at least one PDF.")

    max_bytes = max_file_size_mb * 1_048_576
    extracted: list[PageDocument] = []
    invalid_messages: list[str] = []
    for upload in files:
        name = str(getattr(upload, "name", "uploaded.pdf"))
        if not name.lower().endswith(".pdf"):
            invalid_messages.append(f"{name}: only PDF files are allowed.")
            continue
        data = _read_upload_bytes(upload)
        if not data:
            invalid_messages.append(f"{name}: the file is empty.")
            continue
        if len(data) > max_bytes:
            invalid_messages.append(f"{name}: exceeds the {max_file_size_mb} MB limit.")
            continue
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if reader.is_encrypted and reader.decrypt("") == 0:
                invalid_messages.append(f"{name}: encrypted PDFs are not supported.")
                continue
        except Exception as exc:
            invalid_messages.append(f"{name}: could not be opened as a PDF ({exc}).")
            continue

        file_hash = hashlib.sha256(data).hexdigest()[:16]
        text_page_count = 0
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = _clean_text(page.extract_text() or "")
            except Exception:
                text = ""
            if not text:
                continue
            text_page_count += 1
            extracted.append(
                PageDocument(
                    text=text,
                    metadata={"document_name": name, "page_number": page_number, "file_hash": file_hash},
                )
            )
        if text_page_count == 0:
            invalid_messages.append(f"{name}: no extractable text was found. This may be a scanned PDF requiring OCR.")

    if invalid_messages and not extracted:
        raise PDFValidationError(" ".join(invalid_messages))
    if not extracted:
        raise PDFValidationError("No readable PDF text was found.")
    return extracted
