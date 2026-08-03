"""
Task 5 - Phase 1: Deep Text Preprocessing Foundation
Features: Lowercasing, Regex cleaning, Stopword removal, Stemming & Lemmatization.
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------- ⭐ STAR: NLTK Data Download (Ek baar karna padta hai) ----------
# Ye line ensure karta hai ki 'stopwords', 'punkt' (tokenizer) aur 'wordnet' (lemmatizer) download ho jayein.
# Agar ye nahi kiya, toh 'LookupError' aata hai.
# Online environment mein 'nltk.download()' GUI khulta hai, isliye hum specific downloads use kar rahe hain.
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
# -------------------------

def clean_basic_regex(text):

    """
    ⭐ STAR: Regular Expressions (re module) - Text ko ganda se saaf karna.
    Order matters: Pehle URLs hatao, phir mentions/hashtags, phir punctuation.
    """

    text = text.lower()

    text = re.sub(r'https?://\s+|www\.\S+','',text)

    # 3. ⭐ STAR: Remove Mentions (@username) aur Hashtags (#topic)
    # '@' se shuru hone wala word, aur '#' se shuru hone wala word hatao.   
    text = re.sub(r'@\w+|#\w+','',text)

    # 4. Remove Extra Punctuation & Numbers (Sirf alphabets aur spaces rakho)
    # [^a-zA-Z\s] matlab: 'a-z, A-Z, aur space' ke alawa sab kuch hatao.
    # ⭐ STAR: re.sub(pattern, replacement, string)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Remove extra spaces (2 ya zyada spaces ko 1 space mein badlo)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_and_filter(text):
    
    # ⭐ STAR: word_tokenize() - NLTK ka standard tokenizer.
    # Ye punctuation aur words ko alag karta hai (lekin humne pehle punctuation hata diya, toh sirf words aayenge).    
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    filtered_words = [word for word in tokens if word.lower() not in stop_words and len(word)>1]

    return filtered_words

def apply_stemming(tokens):
    """
    ⭐ STAR: PorterStemmer - Words ki roots nikaalta hai (e.g., running -> run, studies -> studi).
    Ye grammatical rules par kaam karta hai. Fast hai, lekin kabhi kabhi exact dictionary word nahi hota.
    """
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]
    return stemmed

def apply_lemmatization(tokens):
    """
    ⭐ STAR: WordNetLemmatizer - Words ko dictionary (WordNet) se match kar ke root (lemma) nikaalta hai.
    e.g., 'running' -> 'run', 'better' -> 'good'. Isme POS (Part of Speech) bhi specify kar sakte hain.
    """
    lemmatizer = WordNetLemmatizer()
    # Default POS: 'n' (noun). Better results ke liye hum 'v' (verb) bhi try kar sakte hain.
    lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return lemmatized


def full_pipeline(raw_text,use_lemmatization=False):
    """
    Raw text ko le kar processed output deta hai.
    use_lemmatization=True -> Lemmatization, False -> Stemming (Fast).
    """
    print("\n" + "="*60)
    print(f"📝 Original Text: {raw_text[:100]}...")  # Sirf pehle 100 chars dikhao

    # Step 1: Regex
    cleaned = clean_basic_regex(raw_text)
    print(f"🧹 After Regex: {cleaned}")

    # Step 2: Tokenize & Stopword removal
    tokens = tokenize_and_filter(cleaned)
    print(f"✂️ Tokens (after stopwords): {tokens}")
    
    # Step 3: Stemming / Lemmatization
    if use_lemmatization:
        final_tokens = apply_lemmatization(tokens)
        print(f"📖 Lemmatized Output: {final_tokens}")
    else:
        final_tokens = apply_stemming(tokens)
        print(f"⚡ Stemmed Output: {final_tokens}")
    
    return final_tokens


if __name__=="__main__":
    # ⭐ STAR: `__name__` == "__main__" ensures ye tabhi chale jab hum file ko directly run karein.

    # Example 1: Dirty Tweet
    tweet = "RT @john_doe: I LOVED the new #AI2025 conference! Check out https://t.co/xyz123 for more!!! 😍😍"
    full_pipeline(tweet, use_lemmatization=False)
    
    # Example 2: Formal Email
    email = "Regarding the meeting, we will be discussing the implementation of the new policies. Please revert back by EOD."
    full_pipeline(email, use_lemmatization=True)

    # Example 3: Complex Words
    complex_text = "The children were playing beautifully and running quickly towards the park."
    full_pipeline(complex_text, use_lemmatization=False)  # Stemming: children -> children? Actually 'children' stem 'children' hi rahega.
    print("\n" + "="*60)
    print("✅ Phase 1 Practice Complete! Try changing the `raw_text` variable to test your own inputs.")