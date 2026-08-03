# Chunk 1: Word2Vec - Training (Meaning Capture)
# Note: Gensim ko list-of-lists (tokenized) chahiye, raw string nahi.
from requests import models
from sklearn.feature_extraction import _stop_words
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

sentences = [
    "I love machine learning and deep learning",
    "Machine learning is fun",
    "Deep learning is a subset of machine learning",
    "I love learning new things"
]

tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
print("Tokenized Sentences:", tokenized_sentences)


print("📝 Tokenized Data:", tokenized_sentences)


# Chunk 2: Word Embeddings - Similarity & Maths

print(f"")
try:
    model = Word2Vec(sentences=tokenized_sentences, vector_size=10, window=3, min_count=1, epochs=10)
    similar_words = model.wv.most_similar("learning", topn=3)
    for word, score in similar_words:
        print(f"{word}:{score:.4f}")
except KeyError:
    print(f"Vocabulary not found")
    similar_words = model.wv.most_similar("deep", topn=2)
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")


# 2. ⭐ STAR: Vector Arithmetic (King - Man + Woman wala concept)
# Humein 'learning' aur 'machine' ka relation samajhna hai.
# Maan lo learning - machine + love = ?
try:
    result_vec = model.wv.most_similar(
        positive=["learning", "love"], negative=["machine"])
    print(f"word after calculation ['learning' - 'machine' + 'love']:")
    for word, score in result_vec:
        print(f"{word} : {score:.4f}")
except KeyError:
    print("\n⚠️ Skip vector math (vocabulary too small).")
