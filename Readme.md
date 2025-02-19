
# ReviewRover - Personalized Reading Assistant

## Description
ReviewRover is a text similarity engine that recommends content based on user input. Using TF-IDF and Word Embedding techniques (Word2Vec), it provides relevant article, review, or comment recommendations.

## Dataset
- Product Reviews (Amazon) https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews

## Tasks
### Task 1: Data Preprocessing and Exploration
- Remove URLs and special characters
- Lowercase the text
- Stopword removal
- Stemming or Lemmatization
- Tokenization

### Task 2: Implementing TF-IDF Based Similarity
- Generate a TF-IDF matrix
- Compute cosine similarity with TF-IDF vectors
- Recommend top 5 similar texts

### Task 3: Implementing Word Embeddings for Similarity
- Train Word2Vec (CBOW and Skip-gram)
- Convert documents and query into word embeddings
- Compute cosine similarity using Word2Vec
- Compare top 5 recommendations from both CBOW and Skip-gram models


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abdul-Moiz-Shehzad/ReviewRover.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Running the App
   ```python
   streamlit run app.py
   ```
