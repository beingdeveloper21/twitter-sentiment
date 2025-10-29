import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def basic_clean(text: str) -> str:
    """Tweet normalization that keeps useful signals (emojis, punctuation emphasis).
    - Replace URLs, mentions with placeholders
    - Strip leading # but keep hashtag text
    - Normalize elongated sequences (sooooo -> sooo)
    - Lowercase and collapse whitespace
    """
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " USER ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", lambda m: m.group(0)[1:], text)
    # normalize elongated characters (3+ occurrences -> 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # do NOT strip punctuation/emojis entirely; char n-grams will capture them
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_sentiment140(csv_path: Path) -> pd.DataFrame:
    # Sentiment140 has 6 columns without header
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(csv_path, names=cols, encoding='ISO-8859-1')
    # Map 0->0 (negative), 4->1 (positive)
    df['target'] = df['target'].replace({4: 1})
    return df[['text', 'target']]


def build_pipeline(
    model: str = 'svm',
    min_df: int = 3,
    max_df: float = 0.98,
    word_ngram_max: int = 2,
    max_features: int | None = 400000,
    C: float = 1.0,
    char_analyzer: str = 'char',
    char_ngram_min: int = 2,
    char_ngram_max: int = 6,
    char_max_features_ratio: float = 0.5,
):
    """Construct a strong baseline using word + character n-grams.
    - Word TF-IDF (1..word_ngram_max)
    - Char TF-IDF (char_ngram_min..char_ngram_max) to capture emojis/punctuation
    - LinearSVC (default) or LogisticRegression
    """
    word_vect = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, word_ngram_max),
        sublinear_tf=True,
        max_features=max_features,
        preprocessor=basic_clean,
    )
    char_vect = TfidfVectorizer(
        analyzer=char_analyzer,  # 'char' includes emojis/punctuation
        ngram_range=(char_ngram_min, char_ngram_max),
        min_df=3,
        sublinear_tf=True,
        max_features=int(max_features * char_max_features_ratio) if max_features else None,
        preprocessor=basic_clean,
    )

    features = FeatureUnion([
        ('word', word_vect),
        ('char', char_vect),
    ])

    if model == 'logreg':
        clf = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='saga',
            n_jobs=None,
            C=C,
        )
    else:
        clf = LinearSVC(C=C, class_weight='balanced')

    return Pipeline([
        ('tfidf', features),
        ('clf', clf),
    ])


def train(
    data_csv: Path,
    out_path: Path,
    test_size: float = 0.2,
    random_state: int = 2,
    sample: int | None = None,
    model: str = 'svm',
    min_df: int = 3,
    max_df: float = 0.98,
    word_ngram_max: int = 2,
    max_features: int | None = 400000,
    C: float = 1.0,
    tune: bool = False,
    tune_iterations: int = 15,
    cv: int = 3,
):
    df = load_sentiment140(data_csv)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=random_state).reset_index(drop=True)

    X = df['text'].astype(str).map(basic_clean)
    y = df['target'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipe = build_pipeline(
        model=model,
        min_df=min_df,
        max_df=max_df,
        word_ngram_max=word_ngram_max,
        max_features=max_features,
        C=C,
    )

    best_params = {}
    if tune:
        # Parameter ranges tailored for speed and gains
        param_distributions = {
            'tfidf__word__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__word__min_df': [3, 5, 10],
            'tfidf__word__max_df': [0.9, 0.95, 0.99],
            'tfidf__word__max_features': [200_000, 300_000, 400_000],
            'tfidf__char__ngram_range': [(2, 5), (3, 5), (3, 6)],
            'tfidf__char__max_features': [100_000, 150_000, 200_000],
            'clf__C': [0.5, 1.0, 1.5, 2.0, 3.0],
        }
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions,
            n_iter=tune_iterations,
            cv=cv_splitter,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1,
            random_state=random_state,
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        best_params = search.best_params_
    else:
        pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    # Extract vocab sizes from union components for logging
    vocab_sizes = {}
    try:
        word_vocab = pipe.named_steps['tfidf'].transformer_list[0][1].vocabulary_
        char_vocab = pipe.named_steps['tfidf'].transformer_list[1][1].vocabulary_
        vocab_sizes = {
            'word_vocab_size': int(len(word_vocab)) if word_vocab is not None else None,
            'char_vocab_size': int(len(char_vocab)) if char_vocab is not None else None,
        }
    except Exception:
        pass

    metrics = {
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'test_accuracy': float(accuracy_score(y_test, y_pred_test)),
        'test_f1_macro': float(f1_score(y_test, y_pred_test, average='macro')),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        **vocab_sizes,
        'model': pipe.named_steps['clf'].__class__.__name__,
        'tuned': bool(tune),
        'best_params': best_params,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(pipe, f)

    metrics_path = out_path.with_suffix('.metrics.json')
    pd.Series(metrics).to_json(metrics_path, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis pipeline (enhanced TF-IDF + Linear SVM by default).')
    parser.add_argument('--data', type=Path, required=True, help='Path to Sentiment140 CSV (training.1600000.processed.noemoticon.csv)')
    parser.add_argument('--out', type=Path, default=Path('sentiment_pipeline.pkl'), help='Output path for pickled pipeline')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=2)
    parser.add_argument('--sample', type=int, default=None, help='Optional sample size for faster experimentation')
    parser.add_argument('--model', choices=['svm', 'logreg'], default='svm')
    parser.add_argument('--min-df', type=int, default=3)
    parser.add_argument('--max-df', type=float, default=0.98)
    parser.add_argument('--word-ngram-max', type=int, default=2)
    parser.add_argument('--max-features', type=int, default=400000)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--tune', action='store_true', help='Run RandomizedSearchCV for better hyperparameters')
    parser.add_argument('--tune-iterations', type=int, default=15)
    parser.add_argument('--cv', type=int, default=3)
    args = parser.parse_args()

    metrics = train(
        args.data,
        args.out,
        args.test_size,
        args.random_state,
        args.sample,
        args.model,
        args.min_df,
        args.max_df,
        args.word_ngram_max,
        args.max_features,
        args.C,
        args.tune,
        args.tune_iterations,
        args.cv,
    )
    print(metrics)


if __name__ == '__main__':
    main()
