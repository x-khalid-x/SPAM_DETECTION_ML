import os
import joblib
import streamlit as st


MODEL_PATH = "models/spam_pipeline.pkl"

st.set_page_config(page_title="Spam Detection", page_icon="📩")
st.title("📩 Détection de spam")
st.write("Colle un message et le modèle prédit s'il est **SPAM** ou **HAM**.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Modèle introuvable. Lance d'abord : `python src/train.py`")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

msg = st.text_area("Message", height=160, placeholder="Ex: Congratulations! You've won...")

if st.button("Prédire"):
    if not msg.strip():
        st.warning("Veuillez entrer un message.")
    else:
        pred = model.predict([msg])[0]
        label = "🚫 SPAM" if pred == 1 else "✅ HAM"
        st.subheader(f"Résultat : {label}")

        # 1) Si le modèle a predict_proba (pas le cas de LinearSVC), on affiche la proba
        if hasattr(model, "predict_proba"):
            proba_spam = model.predict_proba([msg])[0][1]
            st.write(f"Probabilité spam : **{proba_spam:.2%}**")

        # 2) Sinon (cas LinearSVC), on affiche un score de confiance (decision_function)
        elif hasattr(model, "decision_function"):
            score = model.decision_function([msg])[0]
            st.caption("Ce modèle (LinearSVC) ne fournit pas de probabilité, mais un **score de décision**.")
            st.write(f"Score de décision : **{score:.3f}**")
            st.progress(min(1.0, abs(float(score)) / 5.0))  # barre indicative

            if score >= 0:
                st.write("Interprétation : score positif → plutôt **SPAM**")
            else:
                st.write("Interprétation : score négatif → plutôt **HAM**")

        else:
            st.caption("Ce modèle ne fournit ni probabilité ni score de décision.")

