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
from sklearn.svm import LinearSVC

old_df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\News Category Dataset.csv" , encoding = "latin1")
new_df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\News Category Dataset 2.csv")

old_df = old_df.drop_duplicates()

# print(old_df["SECTION"].value_counts())


new_df = new_df.drop(["headline" , "links" , "keywords"] , axis = 1)


new_df = new_df.drop_duplicates()

# print(new_df["category"].value_counts())

requiredcategories = ["POLITICS" , "BUSINESS" , "SPORTS" , "WORLD NEWS"]
new_df = new_df[new_df['category'].isin(requiredcategories)]
# print(new_df["category"].value_counts())

# print(new_df.columns)

new_df = new_df.rename (columns = {"category" : "SECTION" , "short_description" : "STORY"})

mapping = { "POLITICS" : 0, "BUSINESS" : 3, "SPORTS" : 4 , "WORLD NEWS" : 5}
new_df["SECTION"] = new_df["SECTION"].map(mapping)


df = pd.concat([old_df , new_df] , ignore_index = True)

# print(df["SECTION"].value_counts())


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


x = df.drop("SECTION" , axis = 1)
y = df["SECTION"]

# print(df["SECTION"].value_counts())


X_train , X_test , y_train , y_test = train_test_split(df["news_description"] , df["SECTION"], test_size = 0.2 , random_state = 42)

tfidf = TfidfVectorizer(ngram_range = (1,3) , max_features = 20000 , min_df = 2)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

ros = RandomOverSampler(random_state=42)
X_train_resampled , y_train_resampled = ros.fit_resample(X_train_tfidf , y_train)

lr = LogisticRegression(max_iter = 2000 , class_weight = "balanced")
lr.fit(X_train_resampled , y_train_resampled)
y_pred = lr.predict(X_test_tfidf)


joblib.dump(lr, "News_Category_lr_model.pkl")
joblib.dump(tfidf, "News_Category_tfidf.pkl")

accuracy = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test , y_pred)
classify = classification_report(y_test , y_pred)
# print(accuracy)
# print(confused)
# print(classify)


