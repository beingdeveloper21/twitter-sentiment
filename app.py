import streamlit as st
import pickle

# Load your saved model and vectorizer
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
        transformed_tweet = vectorizer.transform([tweet])
        prediction = model.predict(transformed_tweet)[0]
        if prediction == 1:
            st.success("😊 Positive Tweet")
        else:
            st.error("😞 Negative Tweet")
