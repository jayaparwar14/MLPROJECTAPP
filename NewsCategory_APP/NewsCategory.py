import re 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
import streamlit as st
import os
import nltk

import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

    
nltk.download('stopwords')

# svc = joblib.load("News_Category_svc_model.pkl")
# tfidf = joblib.load("News_Category_tfidf.pkl")

BASE_DIR = os.path.dirname(__file__)

svc = joblib.load(os.path.join(BASE_DIR, "News_Category_svc_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "News_Category_tfidf.pkl"))

lemmitizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text (text) :

    text = text.lower()

    text = re.sub(r"http\S+" , "" , text)

    text = re.sub(r"[^a-zA-Z\s]" , "" , text)

    words = text.split()

    filtered_words = [word for word in words if word not in stop_words]

    lemmitize_words = [lemmitizer.lemmatize(word,pos="v") for word in filtered_words]

    final_text = " ".join(lemmitize_words)

    return final_text

def news_prdict () :

    st.title("📰 News Category Classification System")

    st.subheader("Classify News Articles into Categories Using Machine Learning")

    st.write("""
    Enter a news headline or article below and click the Classify button.

    The machine learning model will analyze the text and identify
    the most appropriate news category.
    """)

    st.header("📝 News Analysis")

    text = st.text_area("Enter News Description " , height = 100)

    cleaned_text = clean_text(text)

    tfidf_newtext = tfidf.transform([cleaned_text])

    prediction = svc.predict(tfidf_newtext)

    if st.button("Predict") :

        if not text.strip() :

            st.warning("⚠️ Please enter a news article or headline for classification.")

        else :

            if prediction == 0:

                st.success("News Category : Politics")
                st.write("🏛️ This article discusses government, public policies, elections, or political events.")

            elif prediction == 1:

                st.success("News Category : Technology")
                st.write("💻 This article is related to technology, innovation, software, AI, or digital trends.")


            elif prediction == 2:

                st.success("News Category : Entertainment")
                st.write("🎬 This article is related to movies, television, celebrities, music, or the entertainment industry.")

            elif prediction == 3:

                st.success("News Category : Business")
                st.write("💼 This article focuses on companies, markets, finance, investments, or economic activities.")

            else:
                st.success("Unknown Category")
                st.info("📄 The content could not be confidently classified into Politics, Technology, Business, Sports, or Entertainment.")
                                                                                                                        

    with st.expander("📖 About this Project"):
        st.markdown("""
        ### 📰 News Category Classification System

        This application uses a Machine Learning model to classify
        news articles into predefined categories.

        **Possible Categories**
        - Business 💼
        - Sports ⚽
        - Technology 💻
        - Entertainment 🎬
        - Politics 🏛️
        - Science 🔬

        **Text Processing Used**
        - Text Cleaning
        - Tokenization
        - Stopword Removal
        - TF-IDF Vectorization

        **Goal**
    
        Automatically categorize news articles based on their content
        and improve information organization.
        """)


news_prdict()
