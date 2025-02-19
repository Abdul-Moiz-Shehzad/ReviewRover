import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from utility.functions import *

st.header("ReviewRover")

with st.spinner("Loading models and data..."):
    @st.cache_resource
    def load_models():
        return {
            "Reviews": joblib.load("Reviews.joblib"),
            "matrix": joblib.load("matrix.joblib"),
        }

data = load_models()
Reviews, matrix = data["Reviews"], data["matrix"]
st.sidebar.header("Method")
method = st.sidebar.radio(" ", ["TFIDF", "Word2Vec"])

if method == "Word2Vec":
    if "w2v_model" not in st.session_state:
        st.session_state.w2v_model = None
    st.sidebar.subheader("Model Type")
    model = st.sidebar.radio(" ", ["CBOW", "Skip-gram"], key="w2v_model")

query = st.text_input("Enter Query")
button = st.button("Search")

if button and query:
    if "results" not in st.session_state:
        st.session_state.results = None
    
    if method == "TFIDF":
        query_preprocess = preprocess_text(query, False)
        similarity = compute_cosine_sim(query, matrix)
    
    elif method == "Word2Vec" and st.session_state.w2v_model:
        similarity = compute_cosine_sim_w2v(cbow if st.session_state.w2v_model == "CBOW" else skip_gram, query, True if st.session_state.w2v_model == "CBOW" else False)
    
    top_documents, top_scores = recommend_top_five(Reviews, similarity)
    st.session_state.results = (top_documents, top_scores, similarity.max())

if "results" in st.session_state and st.session_state.results:
    top_documents, top_scores, max_similarity = st.session_state.results
    st.write("Maximum Similarity =", max_similarity)
    st.subheader("Top Similar Reviews")
    df = pd.DataFrame({"Sentence": top_documents, "Score": top_scores})
    st.write(df)
