import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


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
    return df


def main():
    df = load_data(DATA_PATH)
    X = df["Message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    print("=== Confusion Matrix (format: [[TN FP],[FN TP]]) ===")
    print(cm)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))


if __name__ == "__main__":
    main()
