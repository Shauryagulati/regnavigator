from typing import List, Dict
import re

MAX_CHUNK_LEN = 600
CHUNK_OVERLAP = 100

HEADER_PATTERNS = [
    r"^(ARTICLE|CHAPTER|TITLE|PART)\b.*",
    r"^(Section|SECTION|Sec\.|SEC\.)\b.*",
    r"^(SB|AB)\s*-?\s*\d+",
    r"^Civil Code Section\s+\d+(\.\d+)*",
]


def split_with_offsets(text: str, max_len: int = MAX_CHUNK_LEN, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Simple character-based sliding window with offsets.
    Keeps things robust and predictable for legal PDFs.
    """
    chunks: List[Dict] = []
    text = text or ""
    n = len(text)
    if n == 0:
        return chunks

    start = 0
    while start < n:
        end = min(start + max_len, n)
        chunk_text = text[start:end]
        # strip but keep mapping by adjusting offsets only at ends
        stripped = chunk_text.strip()
        if stripped:
            # compute stripped offsets relative to original text
            leading_ws = len(chunk_text) - len(chunk_text.lstrip())
            trailing_ws = len(chunk_text) - len(chunk_text.rstrip())
            real_start = start + leading_ws
            real_end = end - trailing_ws
            chunks.append({
                "text": text[real_start:real_end],
                "start": real_start,
                "end": real_end,
            })
        if end == n:
            break
        # move window with overlap
        start = end - overlap if end - overlap > 0 else end

    return chunks


def detect_header(passage: str) -> str:
    """
    Heuristic header detector.
    - Looks at the first line of the passage.
    - Treats all-caps or known patterns (ARTICLE, SECTION, SB 53, AB 243, Civil Code Section...) as headers.
    """
    if not passage:
        return ""

    # Take first line
    first_line = passage.strip().splitlines()[0].strip()
    if not first_line:
        return ""

    # If it's very short and mostly punctuation, skip
    if len(first_line) < 3:
        return ""

    # All-caps heuristic (with some letters)
    letters = [c for c in first_line if c.isalpha()]
    if letters and first_line.upper() == first_line:
        return first_line[:200]

    # Known header/bill patterns
    for pat in HEADER_PATTERNS:
        if re.match(pat, first_line):
            return first_line[:200]

    return ""
