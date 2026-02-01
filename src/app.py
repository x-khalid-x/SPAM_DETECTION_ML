import os
import sys
import joblib
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# -------------------------------------------------------------------
# Import preprocess
# -------------------------------------------------------------------
sys.path.append(os.path.dirname(__file__))
from preprocess import normalize_spam_patterns

# -------------------------------------------------------------------
# Paths & Infos
# -------------------------------------------------------------------
MODEL_PATH = "models/spam_pipeline.pkl"
DATA_PATH = "data/spam.csv"

PROJECT_TITLE = "Détection de Spam par Machine Learning"
STUDENT_NAME = "Khalid Chliyahe"
PROGRAM = "Data Science / Machine Learning"
GITHUB_URL = "https://github.com/x-khalid-x/SPAM-DETECTION-ML"
STREAMLIT_URL = "https://TON-APP.streamlit.app"

# -------------------------------------------------------------------
# Streamlit setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Spam Detection", page_icon="📩")
st.title("📩 Détection de spam")
st.write("Colle un message et le modèle prédit s'il est **SPAM** ou **HAM**.")

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------
def score_level(score: float) -> str:
    s = abs(float(score))
    if s < 0.5:
        return "Faible"
    if s < 1.5:
        return "Moyen"
    return "Élevé"

# -------------------------------------------------------------------
# PDF export (PRO)
# -------------------------------------------------------------------
def history_to_pdf_pro(history):
    buffer = BytesIO()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "cover_title",
        parent=styles["Title"],
        alignment=1,
        fontSize=22,
        spaceAfter=20,
    )

    center = ParagraphStyle(
        "center",
        parent=styles["Normal"],
        alignment=1,
        fontSize=12,
        spaceAfter=12,
    )

    small = ParagraphStyle(
        "small",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
    )

    msg_style = ParagraphStyle(
        "msg",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, "Spam Detection – Historique")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    elements = []

    # ---------------- COVER PAGE ----------------
    elements.append(Spacer(1, 3 * cm))
    elements.append(Paragraph(PROJECT_TITLE, title_style))
    elements.append(Paragraph(f"<b>Étudiant :</b> {STUDENT_NAME}", center))
    elements.append(Paragraph(f"<b>Filière :</b> {PROGRAM}", center))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Modèle :</b> LinearSVC + TF-IDF", center))
    elements.append(Paragraph("<b>Métrique :</b> F1-score (classe spam)", center))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>GitHub :</b> {GITHUB_URL}", center))
    elements.append(Paragraph(f"<b>Streamlit :</b> {STREAMLIT_URL}", center))
    elements.append(Spacer(1, 20))
    elements.append(
        Paragraph(
            f"<b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            center,
        )
    )
    elements.append(Spacer(1, 800))  # page break

    # ---------------- SUMMARY ----------------
    total = len(history)
    spam = sum(1 for h in history if h["prediction"] == "SPAM")
    ham = total - spam

    elements.append(Paragraph("Résumé", styles["Heading2"]))
    summary = Table(
        [
            ["Total", total],
            ["SPAM", spam],
            ["HAM", ham],
        ],
        colWidths=[4 * cm, 10 * cm],
    )
    summary.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica"),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(summary)
    elements.append(Spacer(1, 12))

    # ---------------- TABLE ----------------
    elements.append(Paragraph("Détails des prédictions", styles["Heading2"]))

    data = [["Date", "Message", "Prédiction", "Score", "Niveau"]]
    for h in history:
        data.append(
            [
                h["ts"],
                Paragraph(h["message"], msg_style),
                h["prediction"],
                f"{h['score']:.3f}" if h["score"] is not None else "",
                h["level"] or "",
            ]
        )

    table = Table(
        data,
        repeatRows=1,
        colWidths=[3 * cm, 8 * cm, 2.5 * cm, 2 * cm, 2 * cm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph("⚠️ Les prédictions peuvent contenir des erreurs.", small)
    )

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["Category"] = df["Category"].str.lower().str.strip()
    df["label"] = df["Category"].map({"ham": 0, "spam": 1}).astype(int)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_spam_patterns,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )

    model.fit(df["Message"], df["label"])
    joblib.dump(model, MODEL_PATH)
    return model

with st.spinner("Chargement du modèle..."):
    model = load_or_train_model()

# -------------------------------------------------------------------
# Single prediction
# -------------------------------------------------------------------
st.subheader("✅ Test d’un message")

msg = st.text_area("Message")

if st.button("Prédire"):
    if msg.strip():
        pred = model.predict([msg])[0]
        score = model.decision_function([msg])[0]
        level = score_level(score)

        label = "🚫 SPAM" if pred == 1 else "✅ HAM"
        st.subheader(label)
        st.write(f"Score : **{score:.3f}** — Niveau : **{level}**")

        st.session_state.history.insert(
            0,
            {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": msg,
                "prediction": "SPAM" if pred == 1 else "HAM",
                "score": score,
                "level": level,
            },
        )

# -------------------------------------------------------------------
# History + exports
# -------------------------------------------------------------------
st.subheader("🕘 Historique")

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)

    st.download_button(
        "⬇️ Télécharger CSV",
        df_hist.to_csv(index=False).encode(),
        "history.csv",
        "text/csv",
    )

    st.download_button(
        "⬇️ Télécharger PDF (PRO)",
        history_to_pdf_pro(st.session_state.history),
        "history.pdf",
        "application/pdf",
    )

    st.dataframe(df_hist, use_container_width=True)

if st.button("🗑️ Vider l'historique"):
    st.session_state.history = []

