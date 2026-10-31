import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import streamlit as st
import os
import nltk
nltk.download('stopwords')

# lr = joblib.load("SMS_Spam_lr_model.pkl")
# tfidf = joblib.load("SMS_Spam_tfidf.pkl")


BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "SMS_Spam_lr_model.pkl"))

lr = load_model()

@st.cache_resource
def load_vectorizer():
    return joblib.load(os.path.join(BASE_DIR, "SMS_Spam_tfidf.pkl"))

tfidf = load_vectorizer()


# lr = joblib.load(os.path.join(BASE_DIR, "SMS_Spam_lr_model.pkl"))
# tfidf = joblib.load(os.path.join(BASE_DIR, "SMS_Spam_tfidf.pkl"))

def clean_text (text) :

    # Convert text to LowerCase 

    text = text.lower()

    # Remove special characters and digits

    text = re.sub(r"[^a-zA-Z\s]" , "" , text)
    # return text

    # Remove Links

    text = re.sub(r"http\S+" , "" , text)
    # return text

    # Tockenize the text

    # words = word_tokenize(text)
    words = text.split()
    # return words

    # remove stop words 

    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    # return filtered_words

    # Initialize PorterStemmer

    stemmer = PorterStemmer()

    # Perfom Stemming -> convert into base form 

    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    # return stemmed_words

    # Joining the stemmed words back into a single string

    cleaned_text = " ".join(stemmed_words)
    return cleaned_text


## Now creating a deployment function 


def predict_spam () :

    st.title("📩 SMS Spam Detection System")

    st.subheader("Predict Whether a Message is Spam or Ham")

    st.write(
        """
        Enter a text message below and click the Predict button.
        The machine learning model will analyze the message and
        determine whether it is Spam or Ham (Not Spam).
        """
    )

    st.header("✍️ Message Analysis")

    msg = st.text_area("Enter Message to Detect" , height = 50) 

    if st.button("Detect") :


        if not msg.strip() :

            st.warning("⚠️ Please enter a message.")

        else :

            cleaned_msg = clean_text(msg)

            tfidf_msg = tfidf.transform([cleaned_msg])

            prediction = lr.predict(tfidf_msg)

            if prediction[0] == 0 :

                st.success("Not Spam ✅ ")

                st.write("""
                    This message appears to be legitimate and does not contain
                    common spam characteristics according to the trained model.
                    """)

            else :

                st.error("Spam 🚨 ")

                st.write("""
                    This message contains patterns commonly associated with spam,
                    promotional content, scams, or unwanted messages.
                    """)

    with st.expander("📖 About this Project"):
        st.markdown("""
        ### 📩 SMS Spam Detection System

        This project uses a Machine Learning model trained on SMS messages
        to identify spam and legitimate messages.

        **Features Used**
        - Text Cleaning
        - Tokenization
        - Stopword Removal
        - Stemming
        - TF-IDF Vectorization

        **Model Purpose**
        - Detect Spam Messages 🚨
        - Identify Legitimate Messages ✅

        **Technologies**
        - Python
        - Scikit-Learn
        - NLTK
        - Streamlit

        This tool helps users identify potentially harmful or unwanted messages automatically.
        """)    


predict_spam()