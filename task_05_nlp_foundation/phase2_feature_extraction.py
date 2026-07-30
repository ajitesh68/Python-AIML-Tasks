# Chunk 1: Data Setup
from numpy import sort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# CountVectorizer: Simple counting.
# TfidfVectorizer: Smart counting.


corpus = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document?"
]


print("document length:", len(corpus))
for i, doc in enumerate(corpus):
    print(f"Doc{i+1}:{doc}")


# Chunk 2: CountVectorizer (Sirf Counts)

# CountVectorizer(): ⭐ STAR. Isne abhi kuch nahi kiya, bas object banaya.
vectorizer = CountVectorizer()

x_counts = vectorizer.fit_transform(corpus)
# fit_transform(corpus): ⭐ STAR.
# Step 1 (fit): Isne saare documents padhe, aur ek Dictionary (Vocabulary) banayi (unique words).
# Step 2 (transform): Isne har document ko lekar, vocabulary ke hisaab se ek Row banayi.
#  Agar word hai toh count (1), nahi hai toh 0.

# vectorizer.get_feature_names_out(): Ye vocabulary array return karta hai.
print("Vocabulary:", vectorizer.get_feature_names_out())
print("\nEncoded vector:", x_counts)


# 4. Dense Matrix mein convert karke dekho (Taaki dikhe)
print("\n📊 Document-Term Matrix (Rows=Docs, Cols=Words):")
print(X_counts.toarray())


# Chunk 3: TF-IDF (Text to Numbers - The Smart Way)
print("\n🧠 --- TF-IDF VECTORIZER (Smart Bag of Words) ---")

tfidf = TfidfVectorizer()

X_tfidf = tfidf.fit_transform(corpus)

print("vocabulary : ", tfidf.get_feature_names_out())

# 4. Dense Matrix mein convert karo
print("TFIDF Matrix (Rows = Docs, Cols = Words)")
print(X_tfidf.toarray())

print(f"Feature Importannce (Word-> TFIDF Score)for Document 1:")
word_importance = list(
    zip(tfidf.get_feature_names_out(), X_tfidf.toarray()[0]))

for word, score in sorted(word_importance, key=lambda x: x[1], reverse=True)
    print(f"{word} : {score:.4f}")



corpus = [
    "I love machine learning and deep learning",
    "Machine learning is fun and challenging"
]

# ⭐ STAR: ngram_range(1,2) means -> Single words (1) aur 2-word pairs (2) dono ko capture karo.
vectorizer = CountVectorizer(ngram_range=(1,2))

X = vectorizer.fit_transform(corpus)


print(f"Ngram vocabulary:"vectorizer.get_feature_names_out())
print(X.toarray())



