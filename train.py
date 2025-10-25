import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def basic_clean(text: str) -> str:
    # Normalize URLs, mentions, hashtags, numbers, and whitespace
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " USER ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", lambda m: m.group(0)[1:], text)  # keep hashtag word
    text = re.sub(r"[^A-Za-z\s]", " ", text)  # remove punctuation/emojis
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_sentiment140(csv_path: Path) -> pd.DataFrame:
    # Sentiment140 has 6 columns without header
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(csv_path, names=cols, encoding='ISO-8859-1')
    # Map 0->0 (negative), 4->1 (positive)
    df['target'] = df['target'].replace({4: 1})
    return df[['text', 'target']]


def train(
    data_csv: Path,
    out_path: Path,
    test_size: float = 0.2,
    random_state: int = 2,
    sample: int | None = None,
):
    df = load_sentiment140(data_csv)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=random_state).reset_index(drop=True)

    X = df['text'].astype(str).map(basic_clean)
    y = df['target'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        solver='saga',
        n_jobs=None,
    )
    pipe = make_pipeline(vectorizer, clf)
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    metrics = {
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'test_accuracy': float(accuracy_score(y_test, y_pred_test)),
        'test_f1_macro': float(f1_score(y_test, y_pred_test, average='macro')),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'vocab_size': int(len(vectorizer.vocabulary_)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(pipe, f)

    metrics_path = out_path.with_suffix('.metrics.json')
    pd.Series(metrics).to_json(metrics_path, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis pipeline (TF-IDF + Logistic Regression).')
    parser.add_argument('--data', type=Path, required=True, help='Path to Sentiment140 CSV (training.1600000.processed.noemoticon.csv)')
    parser.add_argument('--out', type=Path, default=Path('sentiment_pipeline.pkl'), help='Output path for pickled pipeline')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=2)
    parser.add_argument('--sample', type=int, default=None, help='Optional sample size for faster experimentation')
    args = parser.parse_args()

    metrics = train(args.data, args.out, args.test_size, args.random_state, args.sample)
    print(metrics)


if __name__ == '__main__':
    main()