import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Production-grade text sanitization and normalization:
    1. Unicode normalization (NFKC)
    2. Replace encoding artifacts (\\ufffd) and standardize smart quotes/dashes
    3. Strip non-printable control characters
    4. Remove common recurring page header/footer patterns
    5. Re-join hyphenated words split across line breaks (e.g. 'govern-\\ning' -> 'governing')
    6. Normalize redundant horizontal and vertical whitespace
    """
    if not text:
        return ""
        
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Fix replacement chars & standardize smart punctuation
    text = text.replace("\ufffd", "'")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    
    # 3. Strip non-printable control characters (keep newline and tab)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ch >= " ")
    
    # 4. Remove recurring page header patterns (e.g. 'Page X | ABC Company...')
    text = re.sub(r"Page\s+\d+\s*\|\s*ABC Company[^\n]*\n?", "", text, flags=re.IGNORECASE)
    
    # 5. Fix hyphenated words broken across lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    
    # 6. Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    
    # 7. Collapse 3+ consecutive newlines into 2 (preserving paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()
