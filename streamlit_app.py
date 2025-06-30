import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.success("✅ The review is **Positive**.")
        else:
            st.error("❌ The review is **Negative**.")
