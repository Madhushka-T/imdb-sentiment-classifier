import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
vader = SentimentIntensityAnalyzer()
# Load model and vectorizer
model = joblib.load('xgb_imdb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit App UI
st.title("🎬 IMDb Review Sentiment Classifier")

st.write("Enter a movie review below to see if it's **Positive** or **Negative** 🎭")

review_input = st.text_area("Enter your review here", height=150)

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = preprocess_text(review_input)
        word_count = len(cleaned.split())

        # Use VADER if sentence is too short
        if word_count <= 5:
            vader_score = vader.polarity_scores(cleaned)['compound']
            if vader_score >= 0.05:
                st.success("✅ The review is **Positive.**")
            elif vader_score <= -0.05:
                st.error("❌ The review is **Negative.**")
            else:
                st.info("😐 The review seems **Neutral.**")
        else:
            # Use your ML model for longer reviews
            transformed = vectorizer.transform([cleaned])
            prediction = model.predict(transformed)[0]

            if prediction == 1:
                st.success("✅ The review is **Positive.** (via XGBoost)")
            else:
                st.error("❌ The review is **Negative.** (via XGBoost)")
