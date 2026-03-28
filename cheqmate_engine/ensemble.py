# ensemble.py
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ModelTrainer")

class AIEnsemble:
    """
    Weighted soft-voting ensemble of three independent classifiers.
    Defined here (not in __main__) so joblib can deserialize it correctly.
    """

    def __init__(self, weights=(2, 2, 1)):
        raw          = np.array(weights, dtype=float)
        self.weights = raw / raw.sum()
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _make_word_model():
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=25_000,
                sublinear_tf=True, min_df=3,
                strip_accents="unicode", analyzer="word",
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1_000, class_weight="balanced",
                solver="saga", n_jobs=-1,
            )),
        ])

    @staticmethod
    def _make_char_model():
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5),
                max_features=25_000, sublinear_tf=True,
                min_df=5, strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=0.5, max_iter=1_000, class_weight="balanced",
                solver="saga", n_jobs=-1,
            )),
        ])

    @staticmethod
    def _make_ling_model():
        from feature_extractor import LinguisticTransformer
        return Pipeline([
            ("features", LinguisticTransformer()),
            ("scaler",   StandardScaler(with_mean=False)),
            ("clf",      RandomForestClassifier(
                n_estimators=300, max_depth=12,
                random_state=42, n_jobs=-1,
                class_weight="balanced", min_samples_leaf=4,
            )),
        ])

    def fit(self, X, y):
        logger.info("  [1/3] Training Word TF-IDF + LR…")
        self.word_model_ = self._make_word_model()
        self.word_model_.fit(X, y)

        logger.info("  [2/3] Training Char TF-IDF + LR…")
        self.char_model_ = self._make_char_model()
        self.char_model_.fit(X, y)

        logger.info("  [3/3] Training Linguistic Features + RF…")
        self.ling_model_ = self._make_ling_model()
        self.ling_model_.fit(X, y)
        return self

    def predict_proba(self, X):
        p1 = self.word_model_.predict_proba(X)
        p2 = self.char_model_.predict_proba(X)
        p3 = self.ling_model_.predict_proba(X)
        return self.weights[0]*p1 + self.weights[1]*p2 + self.weights[2]*p3

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def estimators_(self):
        return [self.word_model_, self.char_model_, self.ling_model_]

    def feature_importances(self):
        from feature_extractor import LinguisticTransformer
        rf    = self.ling_model_.named_steps["clf"]
        names = LinguisticTransformer().get_feature_names_out()
        imps  = rf.feature_importances_
        return dict(sorted(zip(names, imps), key=lambda x: -x[1]))