import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model = pk.load(open('D:\movie_reviews\model (1).pkl', 'rb'))
scaler = pk.load(open('D:\movie_reviews\scaler.pkl', 'rb'))
review = st.text_input('Enter the movie review')

if st.button('predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write('Negitive review')
    else:
        st.write('Positive review')