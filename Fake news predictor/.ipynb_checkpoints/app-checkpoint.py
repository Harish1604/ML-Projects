import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load models
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Clean text
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    words = re.findall(r'\b\w+\b', text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🕵️‍♂️ Fake News Detector (No NLTK)")
st.write("Enter a news article or headline below:")

user_input = st.text_area("📰 Your news text:")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter something to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.success("✅ This looks like **REAL news**.")
        else:
            st.error("🚨 This might be **FAKE news**.")

        st.info(f"🔬 Confidence → Fake: `{prob[0]:.2f}` | Real: `{prob[1]:.2f}`")
