# Chunk 1: Word2Vec - Training (Meaning Capture)
# Note: Gensim ko list-of-lists (tokenized) chahiye, raw string nahi.
from sklearn.feature_extraction import _stop_words
from gensim.models import Word2Vec
import nltk 
from nltk.tokenize import word_tokenzie

sentences = [
    "I love machine learning and deep learning",
    "Machine learning is fun",
    "Deep learning is a subset of machine learning",
    "I love learning new things"
]

tokenized_sentences = [word_tokenzie(sent.lower()) for sent in sentences]
print("Tokenized Sentences:", tokenized_sentences)


print("📝 Tokenized Data:", tokenized_sentences)


# Chunk 2: Word Embeddings - Similarity & Maths

