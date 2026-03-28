"""
feature_extractor.py
--------------------
Extracts 24 hand-crafted linguistic and statistical features from text.
These capture stylistic patterns that reliably differ between human and AI writing.

Key signals:
  - Burstiness (sentence length variance) → low = AI
  - AI-tell phrases (furthermore, moreover, delve…) → high = AI
  - Contraction usage → low = AI
  - Vocabulary richness (TTR, hapax ratio) → AI can be repetitive
  - Passive voice density → AI overuses passive
  - Sentence starter variety → AI repeats "The", "This", "It"
"""

import re
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

# ── Common stopwords (lightweight, no NLTK dependency) ──────────────────────
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
    "not", "no", "so", "if", "as", "up", "out", "about", "into", "than",
    "then", "when", "where", "who", "which", "all", "also", "just", "more",
    "their", "there", "here", "what", "how", "my", "our", "your", "its"
}

# ── Words/phrases that LLMs demonstrably overuse ────────────────────────────
AI_TELLS = {
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "nevertheless", "nonetheless", "in conclusion", "to summarize", "in summary",
    "it is important to note", "it is worth noting", "it should be noted",
    "plays a crucial role", "plays an important role", "a wide range of",
    "a variety of", "in today's world", "in the modern world", "delve",
    "leverage", "utilize", "facilitate", "demonstrate", "ensure", "implement",
    "robust", "seamless", "paradigm", "holistic", "comprehensive", "nuanced",
    "multifaceted", "it is evident", "it is clear", "it can be seen",
    "as mentioned above", "in other words", "that being said", "shed light",
    "in this regard", "it goes without saying", "in the realm of", "underscores",
    "pivotal", "paramount", "groundbreaking", "cutting-edge", "state-of-the-art",
    "showcase", "tapestry", "landscape", "ecosystem", "synergy", "streamline",
    "empower", "foster", "harness", "navigate", "unlock", "elevate"
}

# ── Transition words common in AI writing ────────────────────────────────────
TRANSITIONS = {
    "first", "second", "third", "finally", "additionally", "furthermore",
    "moreover", "however", "nevertheless", "in contrast", "on the other hand",
    "in conclusion", "to conclude", "in summary", "therefore", "thus", "hence",
    "as a result", "consequently", "for instance", "for example", "in addition",
    "on the contrary", "in other words", "to illustrate", "in particular"
}

# ── Contractions humans write, AI tends to avoid ─────────────────────────────
CONTRACTION_PATTERN = re.compile(
    r"\b(?:can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"haven't|hasn't|hadn't|wouldn't|couldn't|shouldn't|it's|i'm|i've|"
    r"i'll|i'd|you're|you've|you'll|they're|we're|he's|she's|that's|"
    r"there's|here's|what's|who's|let's|how's|where's|"
    r"i'm|it'd|that'd|he'd|she'd|we'd|they'd)\b",
    re.IGNORECASE
)

PASSIVE_PATTERN = re.compile(
    r'\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b',
    re.IGNORECASE
)

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
WORD_PATTERN   = re.compile(r'\b[a-zA-Z]+\b')
PARA_SPLIT     = re.compile(r'\n\s*\n')


def extract_features(text: str) -> np.ndarray:
    """
    Extract 24 linguistic/statistical features from raw text.

    Args:
        text: Any text string (prose, OCR output, mixed content).

    Returns:
        np.ndarray of shape (24,) — all float32.
        Returns zeros for empty / too-short inputs.
    """
    if not text or len(text.strip()) < 20:
        return np.zeros(24, dtype=np.float32)

    text_lower = text.lower()

    # ── Tokenise ──────────────────────────────────────────────────────────────
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if len(s.strip()) > 5]
    words     = WORD_PATTERN.findall(text_lower)
    paragraphs = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]

    if not words or not sentences:
        return np.zeros(24, dtype=np.float32)

    n_sent = len(sentences)
    n_words = len(words)

    # ── 1. Sentence-length statistics ─────────────────────────────────────────
    sent_lens   = [len(WORD_PATTERN.findall(s)) for s in sentences]
    avg_sent    = float(np.mean(sent_lens))
    std_sent    = float(np.std(sent_lens)) if n_sent > 1 else 0.0
    # Coefficient of variation — LOW means AI (uniform sentences)
    cv_sent     = std_sent / avg_sent if avg_sent > 0 else 0.0

    # ── 2. Vocabulary features ────────────────────────────────────────────────
    avg_word_len    = float(np.mean([len(w) for w in words]))
    ttr             = len(set(words)) / n_words                       # type-token ratio
    stopword_ratio  = sum(1 for w in words if w in STOPWORDS) / n_words

    word_freq   = Counter(words)
    hapax       = sum(1 for c in word_freq.values() if c == 1)
    hapax_ratio = hapax / len(set(words)) if set(words) else 0.0

    # Repetition score — top-10 words dominating the text (AI can be repetitive)
    top10_count     = sum(c for _, c in word_freq.most_common(10))
    repetition_score = top10_count / n_words

    # ── 3. AI-tell phrases ────────────────────────────────────────────────────
    ai_tell_count   = sum(1 for t in AI_TELLS if t in text_lower)
    ai_tell_density = ai_tell_count / n_sent

    # ── 4. Punctuation densities ──────────────────────────────────────────────
    comma_density       = text.count(',')  / n_sent
    semicolon_density   = text.count(';')  / n_sent
    exclamation_density = text.count('!')  / n_sent
    question_density    = text.count('?')  / n_sent
    # Quotes — human writing often quotes sources informally
    quote_density = (text.count('"') + text.count('\u201c') + text.count('\u201d')) / n_sent

    # ── 5. Sentence structure ─────────────────────────────────────────────────
    # First-word variety: AI repeats the same sentence starters
    first_words = [
        WORD_PATTERN.findall(s)[0].lower()
        for s in sentences if WORD_PATTERN.findall(s)
    ]
    first_word_variety = len(set(first_words)) / len(first_words) if first_words else 0.0
    long_sent_ratio  = sum(1 for l in sent_lens if l > 30) / n_sent
    short_sent_ratio = sum(1 for l in sent_lens if l < 5)  / n_sent

    # ── 6. Contractions & passive voice ──────────────────────────────────────
    contraction_count   = len(CONTRACTION_PATTERN.findall(text))
    contraction_density = contraction_count / n_sent
    passive_count       = len(PASSIVE_PATTERN.findall(text_lower))
    passive_density     = passive_count / n_sent

    # ── 7. Paragraph structure ────────────────────────────────────────────────
    para_count   = float(len(paragraphs))
    avg_para_len = float(np.mean([len(WORD_PATTERN.findall(p)) for p in paragraphs])) if paragraphs else 0.0

    # ── 8. Transition density ─────────────────────────────────────────────────
    transition_count   = sum(1 for t in TRANSITIONS if re.search(r'\b' + t + r'\b', text_lower))
    transition_density = transition_count / n_sent

    # ── Assemble feature vector ───────────────────────────────────────────────
    return np.array([
        avg_sent,           # 0
        std_sent,           # 1
        cv_sent,            # 2  ← burstiness — key AI signal
        avg_word_len,       # 3
        ttr,                # 4
        stopword_ratio,     # 5
        ai_tell_density,    # 6  ← AI-phrase density
        comma_density,      # 7
        semicolon_density,  # 8
        exclamation_density,# 9
        question_density,   # 10
        para_count,         # 11
        avg_para_len,       # 12
        first_word_variety, # 13 ← AI repeats starters
        long_sent_ratio,    # 14
        short_sent_ratio,   # 15
        hapax_ratio,        # 16
        contraction_density,# 17 ← AI avoids contractions
        passive_density,    # 18 ← AI overuses passive
        transition_density, # 19
        quote_density,      # 20
        repetition_score,   # 21
        avg_word_len * ttr, # 22  interaction feature
        cv_sent * (1 - ttr),# 23  interaction: low variety + uniform sentences → AI
    ], dtype=np.float32)


class LinguisticTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that converts raw text into
    the 24-dimensional linguistic feature matrix (as sparse CSR).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        matrix = np.array([extract_features(text) for text in X], dtype=np.float32)
        return csr_matrix(matrix)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "avg_sent_len", "std_sent_len", "cv_sent_len", "avg_word_len",
            "ttr", "stopword_ratio", "ai_tell_density", "comma_density",
            "semicolon_density", "exclamation_density", "question_density",
            "para_count", "avg_para_len", "first_word_variety",
            "long_sent_ratio", "short_sent_ratio", "hapax_ratio",
            "contraction_density", "passive_density", "transition_density",
            "quote_density", "repetition_score", "word_len_x_ttr",
            "cv_x_novelty"
        ])