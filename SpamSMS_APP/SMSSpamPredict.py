import pandas as pd 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
from sklearn.naive_bayes import MultinomialNB , BernoulliNB
from sklearn.svm import SVC
import joblib


# df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\SMS Spam Collection Dataset.csv")
old_df = pd.read_csv(
    "C:\\Users\\jayap\\OneDrive\\Desktop\\SMS Spam Collection Dataset.csv",
    encoding = "latin1"
)
new_df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\ML Python Practice\\custom_sms.csv")
# print(new_df)


# print(df)
# print(df.columns)

## Drop useless columns
old_df.drop(["Unnamed: 2" , "Unnamed: 3" , "Unnamed: 4"] , axis = 1 , inplace = True)

# print(old_df)
# print(old_df.columns)

# new_df["label"] = new_df["label"].str.lower()

## Change column name 

old_df = old_df.rename(

    columns = {
        
        "v1" : "label",
        "v2" : "MSG"
    }
)


new_df = new_df.rename(

    columns = {
        
        "label" : "label",
        "message" : "MSG"
        
    }
)
# print(new_df.columns)


df = pd.concat([old_df,new_df], ignore_index=True)

# print(df.columns)

# print(df.shape)
# print(df["label"].value_counts())


## Check NaN value

# print(df.isnull().sum())

## Check duplicate values and shape of data

# print(df.duplicated().sum())
# print(df.shape)

## Remove duplicates values

df = df.drop_duplicates()
# print(df.duplicated().sum())
# print(df.shape)

# ## Create a function for clean the text 

stop_words = set(stopwords.words("english"))
# Initialize PorterStemmer

stemmer = PorterStemmer()


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

    words = word_tokenize(text)
    # return words

    # remove stop words 

    filtered_words = [word for word in words if word not in stop_words]
    # return filtered_words

    # Perfom Stemming -> convert into base form 

    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    # return stemmed_words

    # Joining the stemmed words back into a single string

    cleaned_text = " ".join(stemmed_words)
    return cleaned_text


## Clean dataset text by using above (clean_text) function

df["SMS"] = df["MSG"].apply(lambda x : clean_text(x))
# print(df)

## Drop old column
df.drop("MSG" , axis = 1 , inplace = True)
# print(df)
# print(df.columns)

v = df["label"].value_counts().plot(kind = "bar")
# print(v)
# plt.show()

## Apply label encoder

le = LabelEncoder()

df["label"] = le.fit_transform(df["label"])
# print(le.classes_)
# print(dict(zip(
#     le.classes_,
#     le.transform(le.classes_)
# )))

## Apply tfidf vectorizer 

tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=5000)

SMS_tfidf = tfidf.fit_transform(df["SMS"])


## Now we apply train test split 

X_train , X_test , y_train , y_test = train_test_split(SMS_tfidf , df["label"] , test_size = 0.2 , random_state = 42)

## Balancing the value of label because yes(0) is more greater than no(1) so the overall result is calculate only on the basis of yes(0)
## So we equilate the value of label

ros = RandomOverSampler()
X_train_resampled , y_train_resampled = ros.fit_resample(X_train , y_train)

lr = LogisticRegression()
lr.fit(X_train_resampled , y_train_resampled)
y_pred = lr.predict(X_test)


joblib.dump(lr, "SMS_Spam_model.pkl")
joblib.dump(tfidf, "SMS_Spam_tfidf.pkl")

accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)

mnb = MultinomialNB()
mnb.fit(X_train_resampled , y_train_resampled)
y_pred = mnb.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)

bnb = BernoulliNB()
bnb.fit(X_train_resampled , y_train_resampled)
y_pred = bnb.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)

svc = SVC()
svc.fit(X_train_resampled , y_train_resampled)
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)


