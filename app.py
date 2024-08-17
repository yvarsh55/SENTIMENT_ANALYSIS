import pandas as pd 
import pickle as pk
import streamlit as st


#st.title("SENTIMENT ANALYSIS ON MOVIE REVIEWS")
st.markdown("<h1 style='color: green;'>SENTIMENT ANALYSIS ON MOVIE REVIEWS</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color: grey;'>Write the movie review below to classify its sentiment</h4>", unsafe_allow_html=True)

#st.subheader("Write the movie review below to classify its sentiment")
st.write("")

st.sidebar.info("This project is made for coderone internship \n Group 32 \n- Kanishka Singh \n - Lavanya Chaudhary \n - Yashu Varshney")

model = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write('Negative Review')
    else:
        st.write('Positive Review')

