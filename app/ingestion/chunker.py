from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.cleaner import clean_text


def get_text_splitter(
    chunk_size: int = 700, 
    chunk_overlap: int = 120
) -> RecursiveCharacterTextSplitter:
    """
    Configure RecursiveCharacterTextSplitter with semantic separators:
    Paragraphs (\\n\\n) -> Lines (\\n) -> Sentences (. , ? , ! ) -> Words ( ).
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )


def chunk_document_pages(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> List[Dict[str, Any]]:
    """
    Cleans and splits text page-by-page, strictly preserving page attribution
    for high-precision enterprise citations.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = []
    global_chunk_idx = 0
    
    for page_info in pages_data:
        raw_text = page_info["text"]
        page_num = page_info["page_number"]
        total_pages = page_info.get("total_pages", 1)
        
        # 1. Clean page text
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
            
        # 2. Split page text
        chunks = splitter.split_text(cleaned)
        
        for c in chunks:
            c_str = c.strip()
            if c_str:
                raw_chunks.append({
                    "content": c_str,
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "chunk_index": global_chunk_idx
                })
                global_chunk_idx += 1
                
    return raw_chunks
