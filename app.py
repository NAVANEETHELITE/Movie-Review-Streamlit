import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

@st.cache_resource
def load_review_model():
    return load_model('MODEL/imdb_model.h5')

@st.cache_resource
def review_word_index():
    return imdb.get_word_index()

@st.cache_resource
def preprocess_review(review):
    review = review.lower().split()
    word_index = review_word_index()
    encoded = [word_index.get(word, 2) + 3 for word in review]
    return sequence.pad_sequences([encoded], maxlen = 500, padding = 'pre')

def get_sentiment(review):
    return load_review_model().predict(preprocess_review(review))[0][0]

st.title("MOVIE REVIEW CLASSIFIER")
try:
    review = st.text_area(label='Enter a review')
    if st.button("SUBMIT"):
        if review:
            prediction = get_sentiment(review)
            if prediction > 0.5:
                st.success(f"Positive : {prediction :.2f}")
            else:
                st.success(f"Negative : {prediction :.2f}")
        else:
            st.write("Please enter a review.")
except Exception as e:
    st.write(e.__class__.__qualname__)