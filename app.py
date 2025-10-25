import streamlit as st
import pickle
from pathlib import Path

# Try to use a serialized end-to-end pipeline if available
PIPELINE_PATH = Path("sentiment_pipeline.pkl")
USE_PIPELINE = PIPELINE_PATH.exists()

if USE_PIPELINE:
    pipeline = pickle.load(open(PIPELINE_PATH, "rb"))
else:
    # Fallback to legacy separate vectorizer + model with NLTK preprocessing
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # Ensure NLTK stopwords are available only if needed
    try:
        _ = stopwords.words('english')
    except Exception:
        nltk.download('stopwords')

    port_stem = PorterStemmer()
    STOP_WORDS = set(stopwords.words('english'))

    def preprocess(text: str) -> str:
        stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(w) for w in stemmed_content if w not in STOP_WORDS]
        return ' '.join(stemmed_content)

    # Load legacy artifacts
    model = pickle.load(open("trained_model.sav", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title
st.title("Twitter Sentiment Analyzer")
st.markdown("A simple ML app that classifies tweets as positive or negative.")

# User input
tweet = st.text_area("Type a tweet here:")

# Predict button
if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet!")
    else:
        if USE_PIPELINE:
            prediction = pipeline.predict([tweet])[0]
        else:
            processed = preprocess(tweet)
            transformed_tweet = vectorizer.transform([processed])
            prediction = model.predict(transformed_tweet)[0]
        if int(prediction) == 1:
            st.success("😊 Positive Tweet")
        else:
            st.error("😞 Negative Tweet")
