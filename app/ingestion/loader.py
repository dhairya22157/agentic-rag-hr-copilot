from pathlib import Path
from typing import List, Dict, Any
import pymupdf
import docx


def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF using PyMuPDF.
    Preserves page numbers (1-indexed) for enterprise citation.
    """
    pages_data = []
    doc = pymupdf.open(file_path)
    total_pages = len(doc)
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        raw_text = page.get_text("text")
        pages_data.append({
            "text": raw_text,
            "page_number": page_idx + 1,
            "total_pages": total_pages
        })
    doc.close()
    return pages_data


def load_docx(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a DOCX file, preserving paragraph and table structure.
    """
    doc = docx.Document(file_path)
    paragraphs = []
    
    # Extract standard paragraphs
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)
            
    # Extract tabular data
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)
                
    full_text = "\n\n".join(paragraphs)
    return [{
        "text": full_text,
        "page_number": 1,
        "total_pages": 1
    }]


def load_txt(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a plain text file with fallback encoding.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
            
    return [{
        "text": text,
        "page_number": 1,
        "total_pages": 1
    }]


def load_document(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Document Loader Router: Automatically dispatches to the correct loader
    based on file extension (.pdf, .docx, .txt).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")
        
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in [".docx", ".doc"]:
        return load_docx(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .pdf, .docx, .txt")
