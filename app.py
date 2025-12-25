import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

label_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

st.set_page_config(page_title="Emotion NLP", layout="centered")
st.title("🧠 Emotion Detection using NLP by Naitik")
st.write("Enter a sentence to predict the emotion")

text = st.text_area("Enter your text:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        emotion = label_map[int(prediction)]

        st.success(f"Predicted Emotion: **{emotion.upper()}**")
