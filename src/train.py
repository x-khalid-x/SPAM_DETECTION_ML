import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from preprocess import normalize_spam_patterns


DATA_PATH = "data/spam.csv"
MODEL_PATH = "models/spam_pipeline.pkl"


def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)

    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Message"] = df["Message"].astype(str)

    df["label"] = df["Category"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    X = df["Message"]
    y = df["label"]
    return X, y


def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_spam_patterns,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", model)
    ])


def get_models():
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "SGDClassifier": SGDClassifier(loss="hinge", class_weight="balanced", random_state=42),
    }


def main():
    X, y = load_data(DATA_PATH)

    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Comparaison des modèles (F1 spam en CV 5-fold) ===")
    scores = {}
    for name, model in models.items():
        pipe = make_pipeline(model)
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()
        scores[name] = f1
        print(f"{name:20s} : {f1:.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\n✅ Meilleur modèle : {best_name} (F1={scores[best_name]:.4f})")

    best_pipeline = make_pipeline(models[best_name])
    best_pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"✅ Pipeline sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
