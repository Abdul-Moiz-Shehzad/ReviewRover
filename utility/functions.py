import re
import string
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tfidf = joblib.load("./tfidf.joblib")
corpus_embedding_cbow = joblib.load("./corpus_embedding_cbow.joblib")
corpus_embedding_skip_gram = joblib.load("./corpus_embedding_skip_gram.joblib")
cbow = joblib.load("./cbow.joblib")
skip_gram = joblib.load("./skip_gram.joblib")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, return_lst=True):
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'http[s]?://\S+|http\s?:\S+', '', text) 
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    words = text.split(' ')
    lst = []
    for word in words:
        if word and word not in stop_words:
            word = lemmatizer.lemmatize(word)
            lst.append(word)
    if return_lst:
        return lst
    else:
        return ' '.join(lst)

def compute_cosine_sim(query, matrix):
    query_processed = preprocess_text(query, False)
    query_mat = tfidf.transform([query_processed])
    similarity = cosine_similarity(query_mat, matrix)
    print(f"Maximum Similarity Score = {similarity.max()}")
    print(f"Average Similarity Score Across all Documents = {similarity.mean()}")
    return similarity.flatten()

def recommend_top_five(reviews, similarity):
    sorted_sim = np.argsort(similarity)[-5:][::-1]
    unique_docs = {}
    for i in sorted_sim:
        doc = reviews[i]
        score = similarity[i]
        if doc not in unique_docs:
            unique_docs[doc] = score
        if len(unique_docs) == 5:
            break
    return list(unique_docs.keys()), list(unique_docs.values())

def get_sentence_embedding(model, sentence):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def compute_cosine_sim_w2v(model, query, cbow=True):
    query_processed = preprocess_text(query, False)
    query_embedding = get_sentence_embedding(model, query_processed)
    if cbow:
        similarity = cosine_similarity([query_embedding], corpus_embedding_cbow)
    else:
        similarity = cosine_similarity([query_embedding], corpus_embedding_skip_gram)
    print(f"Maximum Similarity Score = {similarity.max()}")
    print(f"Average Similarity Score Across all Documents = {similarity.mean()}")
    return similarity.flatten()

def recommend_top_five_w2v(corpus, similarity):
    sorted_sim = np.argsort(similarity)[-5:][::-1]
    unique_docs = {}
    for i in sorted_sim:
        doc = corpus[i]
        sim = similarity[i]
        if doc not in unique_docs:
            unique_docs[doc] = sim
        if len(unique_docs) == 5:
            break
    return list(unique_docs.keys()), list(unique_docs.values())
