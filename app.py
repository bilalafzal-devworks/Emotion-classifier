import streamlit as st
import joblib

# Load model
model = joblib.load("emotion_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Label mapping
label_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

st.title("Emotion/Tone Classifier")

st.write(
    "Enter some text and the model will classify its sentiment."
)

text = st.text_area(
    "Enter your text:"
)

if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        text_vector = vectorizer.transform([text])

        prediction = model.predict(text_vector)[0]

        probabilities = model.predict_proba(text_vector)[0]

        confidence = probabilities[prediction]

        label = label_names[prediction]

        st.success(f"Prediction: {label}")

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )
