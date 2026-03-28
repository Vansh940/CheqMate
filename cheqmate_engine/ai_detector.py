"""
ai_detector.py
--------------
Production AI content detector.

Loads the trained ensemble model (Word TF-IDF + Char TF-IDF + Linguistic RF)
and exposes two public methods:

    detect(text)               → float  (0–100, AI probability %)
    get_detailed_analysis(text)→ dict   (breakdown, flags, confidence)

Special handling
────────────────
  Code blocks     → prose is extracted; code is NOT scored (would give meaningless
                     low scores on a prose-trained model)
  OCR documents   → OCR artifacts are normalised before scoring
  Short texts     → confidence is reduced; scores pulled toward 50 (uncertain)
  Very short texts→ returned as 0.0 (not enough signal)
"""

import os
import sys
import logging
from typing import Dict

import joblib
import numpy as np

# Allow running from project root or from a subdirectory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessor import preprocess
from ensemble import AIEnsemble  # noqa: F401 — needed for joblib deserialization

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("AIDetector")

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_WORDS_HARD       = 20     # below this → always return 0
MIN_WORDS_RELIABLE   = 80     # below this → reduce confidence
CODE_PROSE_THRESHOLD = 0.40   # code_ratio above this → analyze prose only

# ── Verdict bands ─────────────────────────────────────────────────────────────
_VERDICTS = [
    (85, "Almost certainly AI-generated"),
    (70, "Very likely AI-generated"),
    (55, "Likely AI-generated"),
    (45, "Uncertain — could be either"),
    (30, "Likely human-written"),
    (15, "Very likely human-written"),
    ( 0, "Almost certainly human-written"),
]


def _verdict(score: float) -> str:
    for threshold, label in _VERDICTS:
        if score >= threshold:
            return label
    return "Almost certainly human-written"


class AIDetector:
    """
    AI content detector using an ensemble ML model.

    Parameters
    ----------
    model_path : str | None
        Override the default model location.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir   = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "model", "ai_detector_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Train it first:  python training/train_model.py"
            )

        logger.info("Loading AI detection model…")
        self.model = joblib.load(model_path)
        logger.info("AI detection model loaded.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, text: str) -> float:
        """
        Estimate the probability that *text* was AI-generated.

        Returns
        -------
        float
            0.0 – 100.0  (percentage).
            Returns 0.0 for empty / too-short / all-code inputs.
        """
        if not text or not text.strip():
            return 0.0

        meta       = preprocess(text)
        word_count = meta["word_count"]

        if word_count < MIN_WORDS_HARD:
            logger.info(f"Text too short ({word_count} words) — returning 0.")
            return 0.0

        analysis_text = self._choose_analysis_text(meta)
        if not analysis_text or len(analysis_text.split()) < MIN_WORDS_HARD:
            logger.info("Prose portion too short after code removal — returning 0.")
            return 0.0

        try:
            raw_prob = self.model.predict_proba([analysis_text])[0][1]

            # ── Confidence damping for short texts ────────────────────────────
            # Pull probability toward 0.5 (maximum uncertainty) linearly as
            # word_count drops from MIN_WORDS_RELIABLE to MIN_WORDS_HARD.
            if word_count < MIN_WORDS_RELIABLE:
                scale    = (word_count - MIN_WORDS_HARD) / (MIN_WORDS_RELIABLE - MIN_WORDS_HARD)
                raw_prob = 0.5 + (raw_prob - 0.5) * max(scale, 0.0)

            score = round(raw_prob * 100, 2)
            logger.info(
                f"AI score={score:.1f}%  words={word_count}  "
                f"ocr={meta['is_ocr']}  code={meta['is_code']}"
            )
            return score

        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            return 0.0

    def get_detailed_analysis(self, text: str) -> Dict:
        """
        Full analysis with per-model breakdown, flags, and verdict.

        Returns
        -------
        dict with keys:
            ai_probability  float
            verdict         str
            confidence      str   ("High" / "Medium" / "Low" / "None")
            word_count      int
            flags           dict
            model_breakdown dict   (individual model probabilities)
            method          str
            note            str
        """
        if not text or not text.strip():
            return self._empty("Empty input")

        meta       = preprocess(text)
        word_count = meta["word_count"]

        if word_count < MIN_WORDS_HARD:
            return self._empty(f"Text too short ({word_count} words; need {MIN_WORDS_HARD}+)")

        analysis_text    = self._choose_analysis_text(meta)
        prose_only_flag  = (meta["is_code"] or meta["code_ratio"] > CODE_PROSE_THRESHOLD)

        # ── Per-model breakdown ───────────────────────────────────────────────
        breakdown = {}
        try:
            names = ["word_tfidf_lr", "char_tfidf_lr", "linguistic_rf"]
            for name, est in zip(names, self.model.estimators_):
                p = est.predict_proba([analysis_text])[0][1]
                breakdown[name] = round(p * 100, 2)
        except Exception:
            breakdown = {}

        ai_score   = self.detect(text)
        confidence = self._confidence(word_count)

        flags = {
            "contains_code":       meta["is_code"],
            "code_ratio_pct":      round(meta["code_ratio"] * 100, 1),
            "ocr_detected":        meta["is_ocr"],
            "analyzed_prose_only": prose_only_flag,
            "word_count":          word_count,
        }

        return {
            "ai_probability":  ai_score,
            "verdict":         _verdict(ai_score),
            "confidence":      confidence,
            "word_count":      word_count,
            "flags":           flags,
            "model_breakdown": breakdown,
            "method":          "Ensemble: Word TF-IDF + Char TF-IDF + Linguistic RF",
            "note":            self._note(meta, word_count),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _choose_analysis_text(self, meta: dict) -> str:
        """Return the correct text slice to pass to the model."""
        if meta["is_code"] or meta["code_ratio"] > CODE_PROSE_THRESHOLD:
            return meta["prose_text"]
        return meta["cleaned_text"]

    @staticmethod
    def _confidence(word_count: int) -> str:
        if word_count >= 200:
            return "High"
        if word_count >= MIN_WORDS_RELIABLE:
            return "Medium"
        if word_count >= MIN_WORDS_HARD:
            return "Low"
        return "None"

    @staticmethod
    def _note(meta: dict, word_count: int) -> str:
        parts = []
        if meta["is_code"]:
            parts.append("Submission is primarily code — only prose sections were analysed.")
        elif meta["code_ratio"] > 0.15:
            parts.append(
                f"{meta['code_ratio']:.0%} of submission is code and was excluded from scoring."
            )
        if meta["is_ocr"]:
            parts.append("OCR artifacts detected — text was normalised before analysis.")
        if word_count < MIN_WORDS_RELIABLE:
            parts.append(
                f"Short text ({word_count} words). "
                f"Use {MIN_WORDS_RELIABLE}+ words for high-confidence results."
            )
        return " ".join(parts) if parts else "Standard analysis — no issues detected."

    @staticmethod
    def _empty(reason: str) -> Dict:
        return {
            "ai_probability":  0.0,
            "verdict":         "Insufficient text",
            "confidence":      "None",
            "word_count":      0,
            "flags":           {},
            "model_breakdown": {},
            "method":          "N/A",
            "note":            reason,
        }