import os
import sys
import joblib
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Permet d'importer preprocess.py depuis le dossier src/
sys.path.append(os.path.dirname(__file__))
from preprocess import normalize_spam_patterns


MODEL_PATH = "models/spam_pipeline.pkl"
DATA_PATH = "data/spam.csv"

st.set_page_config(page_title="Spam Detection", page_icon="📩")
st.title("📩 Détection de spam")
st.write("Colle un message et le modèle prédit s'il est **SPAM** ou **HAM**.")


@st.cache_resource
def load_or_train_model():
    os.makedirs("models", exist_ok=True)

    # 1) Charger si déjà entraîné
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # 2) Sinon (Streamlit Cloud) entraîner à la volée
    if not os.path.exists(DATA_PATH):
        st.error("Dataset introuvable. Vérifie que `data/spam.csv` est bien sur GitHub.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=str.strip)
    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Message"] = df["Message"].astype(str)
    df["label"] = df["Category"].map({"ham": 0, "spam": 1})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    X = df["Message"]
    y = df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_spam_patterns,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


model = load_or_train_model()

msg = st.text_area("Message", height=160, placeholder="Ex: Congratulations! You've won...")

if st.button("Prédire"):
    if not msg.strip():
        st.warning("Veuillez entrer un message.")
    else:
        pred = model.predict([msg])[0]
        label = "🚫 SPAM" if pred == 1 else "✅ HAM"
        st.subheader(f"Résultat : {label}")

        # LinearSVC -> pas de proba, mais decision_function existe
        if hasattr(model, "decision_function"):
            score = model.decision_function([msg])[0]
            st.caption("LinearSVC ne fournit pas de probabilité, mais un **score de décision**.")
            st.write(f"Score de décision : **{score:.3f}**")
            st.progress(min(1.0, abs(float(score)) / 5.0))

            if score >= 0:
                st.write("Interprétation : score positif → plutôt **SPAM**")
            else:
                st.write("Interprétation : score négatif → plutôt **HAM**")


