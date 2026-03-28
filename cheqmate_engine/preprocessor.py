"""
preprocessor.py
---------------
Handles text normalization before AI detection.

Responsibilities:
  1. Code detection  — submissions with significant code get prose-only analysis
  2. OCR normalization — scanned documents have artifacts that skew scores
  3. Mixed-content splitting — extracts prose from code+prose documents

Why this matters for AI detection accuracy:
  - Code has very different token distributions; feeding it to a prose model
    produces meaningless (often low) scores.
  - OCR noise (merged words, random capitals, garbage chars) looks "unpredictable"
    and falsely pulls the score toward "human-written".
  - Normalizing both inputs before scoring reduces false negatives.
"""

import re
import logging

logger = logging.getLogger("Preprocessor")

# ── Minimum proportions ───────────────────────────────────────────────────────
CODE_LINE_THRESHOLD   = 0.30   # ≥30% code lines → "code submission"
CODE_RATIO_THRESHOLD  = 0.40   # ≥40% words are in code → analyze prose only

# ── Code-line detection patterns ─────────────────────────────────────────────
_CODE_PATTERNS = [
    re.compile(r'def\s+\w+\s*\(.*\)\s*:'),                  # Python function
    re.compile(r'class\s+\w+.*:'),                           # Python/JS class
    re.compile(r'function\s+\w+\s*\(.*\)\s*\{'),            # JS function
    re.compile(r'public\s+(?:static\s+)?\w[\w<>]*\s+\w+\s*\('), # Java/C# method
    re.compile(r'#include\s*[<"].*[>"]'),                    # C/C++ include
    re.compile(r'import\s+[\w.]+(?:\s+as\s+\w+)?\s*;?$'),   # Python / Java import
    re.compile(r'from\s+[\w.]+\s+import\s+[\w*, ]+'),       # Python from-import
    re.compile(r'SELECT\s+.+\s+FROM\s+\w+', re.IGNORECASE), # SQL
    re.compile(r'<\s*[a-zA-Z][^>]*>.*<\s*/[a-zA-Z]'),      # HTML/XML pair
    re.compile(r'^\s*(?:var|let|const)\s+\w+\s*='),         # JS variable
    re.compile(r'^\s*(?:if|for|while)\s*\(.*\)\s*\{?$'),    # C-style control flow
    re.compile(r'^\s*(?:#|//|/\*|\*)\s+\w'),                # Comments
    re.compile(r'^\s*[}\])\s;,]+\s*$'),                     # Closing braces
    re.compile(r'printf\s*\(|cout\s*<<|cin\s*>>'),          # C/C++ IO
    re.compile(r'System\.out\.print|Console\.Write'),        # Java/C# IO
    re.compile(r'@\w+\s*(?:\(.*\))?$'),                     # Decorators/annotations
    re.compile(r'^\s*return\s+.+;?\s*$'),                   # return statements
    re.compile(r'lambda\s+\w+.*:'),                         # Lambda
]

_CODE_KEYWORDS = frozenset({
    'def', 'class', 'import', 'from', 'return', 'yield', 'lambda',
    'elif', 'except', 'finally', 'with', 'assert', 'raise', 'pass',
    'function', 'var', 'let', 'const', 'async', 'await', 'typeof',
    'public', 'private', 'protected', 'static', 'void', 'final',
    'namespace', 'using', 'include', 'template', 'typename',
    'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WHERE', 'JOIN', 'CREATE',
    'printf', 'scanf', 'malloc', 'NULL', 'nullptr',
})

# ── OCR artifact patterns ─────────────────────────────────────────────────────
_MERGED_WORD  = re.compile(r'\w{25,}')            # abnormally long "words"
_MID_CAPS     = re.compile(r'\b[a-z]{2,}[A-Z][a-z]+\b')  # camelCase mid-scan
_GARBAGE_RUNS = re.compile(r'[^\w\s.,!?;:\'"()\-]{3,}')  # non-standard char runs


# ─────────────────────────────────────────────────────────────────────────────
# Code detection
# ─────────────────────────────────────────────────────────────────────────────

def _score_line_as_code(line: str) -> bool:
    """Return True if a single line looks like code."""
    stripped = line.strip()
    if not stripped:
        return False
    # Pattern match
    if any(pat.search(stripped) for pat in _CODE_PATTERNS):
        return True
    # Keyword density (≥2 code keywords in one short line)
    tokens = set(stripped.split())
    if len(tokens & _CODE_KEYWORDS) >= 2:
        return True
    return False


def is_code(text: str) -> bool:
    """
    Return True if the text appears to be primarily source code.
    Threshold: CODE_LINE_THRESHOLD of non-empty lines look like code.
    """
    if not text:
        return False
    lines = text.split('\n')
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False
    code_lines = sum(1 for l in non_empty if _score_line_as_code(l))
    return (code_lines / len(non_empty)) >= CODE_LINE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# OCR detection
# ─────────────────────────────────────────────────────────────────────────────

def is_ocr_text(text: str) -> bool:
    """
    Detect OCR-scanned text by looking for common artifact patterns.
    Returns True when ≥2 indicators are present.
    """
    if not text:
        return False
    score = 0

    if _mid_caps := _MID_CAPS.search(text):
        score += 1
    if _merged := _MERGED_WORD.search(text):
        score += 1
    if _garbage := _GARBAGE_RUNS.search(text):
        score += 1

    # Very short line fragments (column-based scanning)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        short_ratio = sum(1 for l in lines if len(l) < 25) / len(lines)
        if short_ratio > 0.5:
            score += 1

    # Non-printable / high-byte characters
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.05:
        score += 1

    return score >= 2


# ─────────────────────────────────────────────────────────────────────────────
# OCR normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_ocr(text: str) -> str:
    """
    Clean common OCR artifacts so the prose model receives clean input.

    Fixes applied:
      - camelCase splits from merged scan lines → add space before capital
      - Multiple spaces → single space
      - Repeated newlines → paragraph break
      - Non-printable characters → space
      - Curly quotes → straight quotes
      - Digit-in-word 0→O substitution (context-dependent heuristic)
    """
    # Split camelCase merges (e.g., "thedog" → won't help, but "theDog" → "the Dog")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Collapse multiple spaces / tabs
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Normalize paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove non-printable ASCII
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    # Curly quotes → standard
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # Common OCR: digit 0 used instead of letter O inside words
    text = re.sub(r'\b0([a-zA-Z])', r'O\1', text)
    text = re.sub(r'([a-zA-Z])0\b', r'\1O', text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Prose extraction from mixed content
# ─────────────────────────────────────────────────────────────────────────────

def extract_prose(text: str) -> str:
    """
    Remove code sections from mixed documents, returning only prose.

    Handles:
      - Markdown fenced code blocks (``` … ```)
      - Inline code (`…`)
      - 4-space / tab-indented code blocks
    """
    # Remove fenced code blocks
    text = re.sub(r'```[\s\S]*?```', '\n', text)
    text = re.sub(r'~~~[\s\S]*?~~~', '\n', text)

    # Remove inline code
    text = re.sub(r'`[^`\n]+`', '', text)

    # Remove 4-space / tab indented blocks
    lines     = text.split('\n')
    prose_lines = [l for l in lines if not re.match(r'^(?:    |\t)', l)]

    return '\n'.join(prose_lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> dict:
    """
    Analyse and normalise raw submission text.

    Returns a dict:
        cleaned_text   (str)   — OCR-normalised full text
        prose_text     (str)   — code blocks removed
        is_code        (bool)  — majority of text is code
        is_ocr         (bool)  — OCR artifacts detected
        code_ratio     (float) — 0–1, estimated code proportion
        word_count     (int)   — words in cleaned text
    """
    result = {
        "cleaned_text": text or "",
        "prose_text":   text or "",
        "is_code":      False,
        "is_ocr":       False,
        "code_ratio":   0.0,
        "word_count":   0,
    }

    if not text or not text.strip():
        return result

    # ── Code check ────────────────────────────────────────────────────────────
    code_flag          = is_code(text)
    result["is_code"]  = code_flag

    prose              = extract_prose(text)
    result["prose_text"] = prose

    total_words = len(text.split())
    prose_words = len(prose.split())
    result["code_ratio"] = 1.0 - (prose_words / total_words) if total_words else 0.0

    # ── OCR check ─────────────────────────────────────────────────────────────
    ocr_flag          = is_ocr_text(text)
    result["is_ocr"]  = ocr_flag

    cleaned           = normalize_ocr(text) if ocr_flag else text
    result["cleaned_text"] = cleaned
    result["word_count"]   = len(cleaned.split())

    if code_flag:
        logger.info(f"Code submission detected (ratio={result['code_ratio']:.0%})")
    if ocr_flag:
        logger.info("OCR artifacts detected — text normalised")

    return result