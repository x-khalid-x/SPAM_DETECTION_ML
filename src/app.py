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

# --- Historique ---
if "history" not in st.session_state:
    st.session_state.history = []  # liste de dicts


def score_level(score: float) -> str:
    """
    Convertit la valeur absolue du score de décision en niveau lisible.
    Seuils simples (tu peux les ajuster) :
    - < 0.5  : Faible
    - < 1.5  : Moyen
    - >= 1.5 : Élevé
    """
    s = abs(float(score))
    if s < 0.5:
        return "Faible"
    if s < 1.5:
        return "Moyen"
    return "Élevé"


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


with st.spinner("Chargement du modèle..."):
    model = load_or_train_model()


# =========================
# 1) PRÉDICTION SIMPLE
# =========================
st.subheader("✅ Test d’un seul message")

msg = st.text_area("Message", height=140, placeholder="Ex: Congratulations! You've won...")

if st.button("Prédire", key="single_predict"):
    msg_clean = msg.strip()
    if not msg_clean:
        st.warning("Veuillez entrer un message.")
    else:
        pred = int(model.predict([msg_clean])[0])
        label = "🚫 SPAM" if pred == 1 else "✅ HAM"
        st.subheader(f"Résultat : {label}")

        score = None
        level = None

        if hasattr(model, "decision_function"):
            score = float(model.decision_function([msg_clean])[0])
            level = score_level(score)

            st.caption("LinearSVC ne fournit pas de probabilité, mais un **score de décision**.")
            st.write(f"Score de décision : **{score:.3f}**")
            st.info(f"Niveau du score : **{level}**")

            # barre indicative (0 à 1)
            st.progress(min(1.0, abs(score) / 5.0))

            if score >= 0:
                st.write("Interprétation : score positif → plutôt **SPAM**")
            else:
                st.write("Interprétation : score négatif → plutôt **HAM**")

        # Ajouter à l'historique
        st.session_state.history.insert(0, {
            "message": msg_clean,
            "prediction": "SPAM" if pred == 1 else "HAM",
            "score": score,
            "level": level
        })


st.divider()

# =========================
# 2) TEST EN LOT
# =========================
st.subheader("🧪 Test en lot (plusieurs messages)")
st.caption("Colle plusieurs messages : **1 message par ligne**. Un tableau sera affiché.")

batch_text = st.text_area(
    "Messages (1 par ligne)",
    height=180,
    placeholder="hello\nWIN money now!!! http://free.com\nSend your email test@mail.com"
)

cols = st.columns([1, 1, 3])
with cols[0]:
    run_batch = st.button("Tester en lot", key="batch_predict")
with cols[1]:
    clear_batch = st.button("Effacer", key="batch_clear")

if clear_batch:
    batch_text = ""

if run_batch:
    lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
    if not lines:
        st.warning("Ajoute au moins une ligne.")
    else:
        preds = model.predict(lines)

        scores = None
        if hasattr(model, "decision_function"):
            scores = model.decision_function(lines)

        results = []
        for i, m in enumerate(lines):
            pred_i = int(preds[i])
            label_i = "SPAM" if pred_i == 1 else "HAM"

            score_i = float(scores[i]) if scores is not None else None
            level_i = score_level(score_i) if score_i is not None else None

            results.append({
                "Message": m,
                "Prediction": label_i,
                "Score": score_i,
                "Niveau": level_i
            })

            # Ajouter aussi à l'historique
            st.session_state.history.insert(0, {
                "message": m,
                "prediction": label_i,
                "score": score_i,
                "level": level_i
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True)

st.divider()

# =========================
# 3) HISTORIQUE + EXPORT CSV
# =========================
st.subheader("🕘 Historique des prédictions")

col_a, col_b = st.columns([1, 2])

with col_a:
    if st.button("🗑️ Vider l'historique", key="clear_history"):
        st.session_state.history = []
        st.success("Historique vidé.")

with col_b:
    if len(st.session_state.history) > 0:
        hist_df = pd.DataFrame(st.session_state.history)
        # Colonnes plus "propres"
        hist_df = hist_df.rename(columns={
            "message": "Message",
            "prediction": "Prediction",
            "score": "Score",
            "level": "Niveau"
        })
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Télécharger l'historique (CSV)",
            data=csv_bytes,
            file_name="spam_predictions_history.csv",
            mime="text/csv"
        )

if len(st.session_state.history) == 0:
    st.write("Aucune prédiction pour le moment.")
else:
    # Affichage limité
    for i, item in enumerate(st.session_state.history[:15], start=1):
        pred_txt = item.get("prediction", "")
        msg_txt = item.get("message", "")
        score_txt = item.get("score", None)
        level_txt = item.get("level", None)

        st.markdown(f"**{i}. {pred_txt}**  \n{msg_txt}")
        if score_txt is not None:
            extra = f"Score : {float(score_txt):.3f}"
            if level_txt:
                extra += f" — Niveau : {level_txt}"
            st.caption(extra)
        st.divider()

st.caption("⚠️ Ce modèle est un outil d’aide à la décision. Les prédictions peuvent contenir des erreurs.")



