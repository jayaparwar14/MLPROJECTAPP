import pandas as pd 
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report

df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\News Category Dataset.csv" , encoding = "latin1")

# print(df)

# print(df.columns)

# print(df.isnull().sum())

# print(df.duplicated().sum())

df = df.drop_duplicates()
# print(df.duplicated().sum())

lemmitizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text (text) :

    text = text.lower()

    text = re.sub(r"http\S+" , "" , text)

    text = re.sub(r"[^a-zA-Z\s]" , "" , text)

    words = word_tokenize(text)

    filtered_words = [word for word in words if word not in stop_words]

    lemmitize_words = [lemmitizer.lemmatize(word,pos="v") for word in filtered_words]

    final_text = " ".join(lemmitize_words)

    return final_text

df["news_description"] = df["STORY"].apply(lambda x : clean_text(x))
# print(df)

df.drop("STORY" , axis = 1 , inplace = True)

# print(df)

# x = df.drop("SECTION" , axis = 1)
# y = df["SECTION"]

X_train , X_test , y_train , y_test = train_test_split(df["news_description"] , df["SECTION"], test_size = 0.2 , random_state = 42)

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

v = df["SECTION"].value_counts().plot(kind = "bar")

# plt.show()

ros = RandomOverSampler()
X_train_resampled , y_train_resampled = ros.fit_resample(X_train_tfidf , y_train)

# print(type(X_train_resampled))
# print(type(y_train_resampled))

# print(y_train_resampled.value_counts())

lr = LogisticRegression()
lr.fit(X_train_resampled , y_train_resampled)
y_pred = lr.predict(X_test_tfidf)

# joblib.dump(lr, "News_Category_lr_model.pkl")

accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)

svc = SVC()
svc.fit(X_train_resampled , y_train_resampled)

joblib.dump(svc, "News_Category_svc_model.pkl")
joblib.dump(tfidf, "News_Category_tfidf.pkl")

y_pred = svc.predict(X_test_tfidf)
accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)

rfc = RandomForestClassifier()
rfc.fit(X_train_resampled , y_train_resampled)
y_pred = rfc.predict(X_test_tfidf)

# joblib.dump(rfc, "News_Category_rfc_model.pkl")

accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)

