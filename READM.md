# 📩 Spam Detection – Machine Learning Project

## 📌 Description

Ce projet consiste à développer un système de **détection de spam** basé sur le **Machine Learning**, capable de classifier des messages texte en **SPAM** ou **HAM**.  
Le projet couvre l’ensemble du cycle de vie d’un projet ML : exploration, prétraitement, comparaison de modèles, sélection du meilleur modèle, évaluation et déploiement via **Streamlit**.

---

## 🎯 Objectifs

- Analyser un dataset de messages texte (spam / ham)
- Comparer plusieurs modèles de classification
- Sélectionner le meilleur modèle selon le **F1-score (classe spam)**
- Évaluer le modèle avec une **matrice de confusion**
- Déployer une application web interactive avec **Streamlit**
- Utiliser **Git & GitHub** pour la gestion de version

---

## 🗂️ Structure du projet

spam-detection/
│
├── data/
│ └── spam.csv
│
├── notebooks/
│ ├── exploration.ipynb
│ ├── preprocessing.ipynb
│ └── modeling.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ └── app.py
│
├── models/
│ └── spam_pipeline.pkl
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Dataset

- **Nom** : SMS Spam Collection
- **Colonnes** :
  - `Category` : ham / spam
  - `Message` : texte du message
- Dataset **déséquilibré** (la classe spam est minoritaire)

---

## 🔧 Prétraitement des données

Le prétraitement est intégré directement dans un **Pipeline scikit-learn** :

- normalisation du texte
- remplacement de motifs spécifiques au spam :
  - URLs → `__URL__`
  - emails → `__EMAIL__`
  - nombres → `__NUMBER__`
  - montants → `__MONEY__`
- vectorisation **TF-IDF** avec n-grams (1, 2)
- suppression des stopwords
- gestion du déséquilibre avec `class_weight="balanced"`

Cette approche garantit la cohérence entre l’entraînement, l’évaluation et le déploiement.

---

## 🤖 Modèles comparés

Les modèles suivants ont été évalués :

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Classifier (LinearSVC)
- SGDClassifier

### 🔍 Métrique de comparaison

➡️ **F1-score de la classe spam**

Le F1-score est utilisé car le dataset est déséquilibré et cette métrique permet d’équilibrer la précision et le rappel pour la classe la plus critique.

---

## 🏆 Modèle retenu

👉 **LinearSVC**

- Meilleur F1-score moyen en cross-validation
- Très performant pour la classification de texte
- Ne fournit pas de probabilités, mais un **score de décision** est utilisé dans l’application Streamlit pour indiquer la confiance du modèle

---

## 📈 Évaluation

- Split train / test stratifié (80 % / 20 %)
- Métriques utilisées :
  - Precision
  - Recall
  - F1-score
  - Matrice de confusion

---

## ▶️ Commandes d’exécution

### 1️⃣ Créer et activer l’environnement virtuel

```bash
python -m venv .venv
.\.venv\Scripts\activate         # Windows

2️⃣ Installer les dépendances
pip install -r requirements.txt
3️⃣ Entraîner et sélectionner le meilleur modèle
python src/train.py
4️⃣ Évaluer le modèle
python src/evaluate.py
➡️ Affiche :
matrice de confusion
classification report
5️⃣ Lancer l’application Streamlit
streamlit run src/app.py
🌐 Déploiement avec Streamlit Cloud

Pousser le projet sur GitHub

Aller sur Streamlit Community Cloud

Connecter le compte GitHub

Sélectionner le repository

Paramètres :

Main file path : src/app.py

Python version : 3.10+

Déployer l’application
🔁 Git & GitHub
Initialisation du dépôt
git init
git add .
git commit -m "Initial commit - spam detection project"
Connexion au dépôt distant
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
git pushBonnes pratiques de commits

feat: add preprocessing and ML pipeline

feat: compare models using F1-score

feat: add Streamlit app

docs: update README -u origin main

🧠 Technologies utilisées

Python

Pandas

Scikit-learn

Streamlit

Joblib

Git & GitHub
✨ Conclusion
Ce projet présente une approche complète et rigoureuse de la détection de spam, depuis l’analyse des données jusqu’au déploiement d’une application web.L’utilisation de pipelines garantit une solution reproductible, robuste et prête pour la mise en production.
