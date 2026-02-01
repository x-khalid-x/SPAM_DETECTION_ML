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
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
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
PROGRAM = "G.informatique"
GITHUB_URL = "https://github.com/x-khalid-x/SPAM-DETECTION-ML"
STREAMLIT_URL = "https://TON-APP.streamlit.app"  # ← mets ici ton vrai lien streamlit

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

    cover_title = ParagraphStyle(
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
    elements.append(Paragraph(PROJECT_TITLE, cover_title))
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

    # ✅ VRAI saut de page (corrige l'erreur LayoutError)
    elements.append(PageBreak())

    # ---------------- IF EMPTY ----------------
    if not history:
        elements.append(Paragraph("Aucune prédiction disponible.", styles["Normal"]))
        doc.build(elements, onFirstPage=footer, onLaterPages=footer)
        buffer.seek(0)
        return buffer

    # ---------------- SUMMARY ----------------
    total = len(history)
    spam = sum(1 for h in history if h.get("prediction") == "SPAM")
    ham = total - spam
    spam_pct = (spam / total) * 100 if total else 0
    ham_pct = (ham / total) * 100 if total else 0

    elements.append(Paragraph("Résumé", styles["Heading2"]))
    summary = Table(
        [
            ["Total", f"{total}"],
            ["SPAM", f"{spam} ({spam_pct:.1f}%)"],
            ["HAM", f"{ham} ({ham_pct:.1f}%)"],
        ],
        colWidths=[4 * cm, 10 * cm],
    )
    summary.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("FONT", (0, 0), (-1, -1), "Helvetica"),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(summary)
    elements.append(Spacer(1, 12))

    # ---------------- TABLE ----------------
    elements.append(Paragraph("Détails des prédictions", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    data = [["Date", "Message", "Prédiction", "Score", "Niveau"]]
    for h in history:
        ts = h.get("ts", "")
        message = h.get("message", "")
        pred = h.get("prediction", "")
        score = h.get("score", None)
        level = h.get("level", "")

        data.append(
            [
                ts,
                Paragraph(str(message), msg_style),
                pred,
                f"{float(score):.3f}" if score is not None else "",
                level or "",
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
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("⚠️ Les prédictions peuvent contenir des erreurs.", small))

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

    if not os.path.exists(DATA_PATH):
        st.error("Dataset introuvable. Vérifie que `data/spam.csv` est bien sur GitHub.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=str.strip)
    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Message"] = df["Message"].astype(str)
    df["label"] = df["Category"].map({"ham": 0, "spam": 1}).astype(int)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_spam_patterns,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
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

msg = st.text_area("Message", height=140, placeholder="Ex: Congratulations! You've won...")

if st.button("Prédire"):
    if not msg.strip():
        st.warning("Veuillez entrer un message.")
    else:
        pred = int(model.predict([msg])[0])

        score = None
        level = None
        if hasattr(model, "decision_function"):
            score = float(model.decision_function([msg])[0])
            level = score_level(score)

        label_ui = "🚫 SPAM" if pred == 1 else "✅ HAM"
        st.subheader(label_ui)

        if score is not None:
            st.write(f"Score : **{score:.3f}** — Niveau : **{level}**")
            st.progress(min(1.0, abs(score) / 5.0))

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

st.divider()

# -------------------------------------------------------------------
# Batch prediction
# -------------------------------------------------------------------
st.subheader("🧪 Test en lot (plusieurs messages)")
st.caption("Colle plusieurs messages : **1 message par ligne**.")

batch_text = st.text_area(
    "Messages (1 par ligne)",
    height=180,
    placeholder="hello\nWIN money now!!! http://free.com\nSend your email test@mail.com",
)

col1, col2 = st.columns([1, 1])
with col1:
    run_batch = st.button("Tester en lot")
with col2:
    clear_batch = st.button("Effacer")

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

        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = []

        for i, m in enumerate(lines):
            pred_i = int(preds[i])
            label_i = "SPAM" if pred_i == 1 else "HAM"

            score_i = float(scores[i]) if scores is not None else None
            level_i = score_level(score_i) if score_i is not None else None

            results.append(
                {"Message": m, "Prediction": label_i, "Score": score_i, "Niveau": level_i}
            )

            st.session_state.history.insert(
                0,
                {
                    "ts": now_ts,
                    "message": m,
                    "prediction": label_i,
                    "score": score_i,
                    "level": level_i,
                },
            )

        st.dataframe(pd.DataFrame(results), use_container_width=True)

st.divider()

# -------------------------------------------------------------------
# History + exports
# -------------------------------------------------------------------
st.subheader("🕘 Historique")

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.download_button(
            "⬇️ Télécharger CSV",
            df_hist.to_csv(index=False).encode("utf-8"),
            "history.csv",
            "text/csv",
        )

    with c2:
        st.download_button(
            "⬇️ Télécharger PDF (PRO)",
            history_to_pdf_pro(st.session_state.history),
            "history.pdf",
            "application/pdf",
        )

    with c3:
        if st.button("🗑️ Vider l'historique"):
            st.session_state.history = []
            st.success("Historique vidé.")

    st.dataframe(df_hist, use_container_width=True)
else:
    st.write("Aucune prédiction pour le moment.")

st.caption("⚠️ Ce modèle est un outil d’aide à la décision. Les prédictions peuvent contenir des erreurs.")


