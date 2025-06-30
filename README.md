# 🎬 IMDb Sentiment Classifier Web App

This is a Streamlit-based web app that predicts whether a given movie review is **positive** or **negative** using NLP and XGBoost.

## 🚀 How it Works

- Preprocesses user input using `nltk`
- Transforms text with TF-IDF
- Loads a trained `XGBoost` model (`xgb_imdb_model.pkl`)
- Displays result as Positive ✅ or Negative ❌

## 📂 Files

- `streamlit_app.py` – main Streamlit web app
- `xgb_imdb_model.pkl` – trained XGBoost model
- `tfidf_vectorizer.pkl` – fitted TF-IDF vectorizer
- `requirements.txt` – Python dependencies
- `README.md` – this file

## 🌐 Deployment

You can deploy it free on [Streamlit Cloud](https://streamlit.io/cloud) for a live demo.

## ✅ Example

Enter a review like:

> _"I absolutely loved this movie. The acting was brilliant!"_

And get:

> **✅ The review is Positive.**

## ✨ Built With

- Python
- Scikit-learn
- XGBoost
- Streamlit
- NLTK
