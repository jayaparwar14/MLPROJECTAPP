import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib


df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\Heart Disease Dataset.csv")

# print(df)

# print(df.columns)

# print(df.isnull().sum())

# print(df.dtypes)


## Apply train test split 

x = df.drop("target" , axis = 1)
y = df["target"]

X_train , X_test , y_train , y_test = train_test_split (x , y , test_size = 0.2 , random_state = 42)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

## Apply standard scale 

scaler = StandardScaler()

X_train_scalled = scaler.fit_transform(X_train)
X_test_scalled = scaler.transform(X_test)


## Train a model random forest classifier

rfc = RandomForestClassifier()

rfc.fit(X_train_scalled , y_train)
y_pred = rfc.predict(X_test_scalled)

joblib.dump(rfc, "HeartDisease_rfc_model.pkl")
joblib.dump(scaler, "HeartDisease_scaler.pkl")

accurate = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test, y_pred)
# print(accurate)
# print(confused)
