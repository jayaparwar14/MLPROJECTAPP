import pandas as pd 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , confusion_matrix
import joblib

df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\Loan Approval Predict Dataset.csv")

# print(df)

# print(df.isnull().sum())

# print(df.dtypes)

## Apply label encoder on columns previous loan gender

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Previous Loan"] = le.fit_transform(df["Previous Loan"])

# mapping = dict(zip(le.classes_, range(len(le.classes_))))
# print(mapping)

## Apply one hot encoder on column education 

# print(df["Education"].value_counts())

ohe = OneHotEncoder(dtype = int , sparse_output = False)
ohe_scalled = ohe.fit_transform(df[["Education" , "Home Onwership" , "Loan Intent"]])

cols = ohe.get_feature_names_out()
# print(cols)

new = pd.DataFrame(data = ohe_scalled , columns = cols)
# print(new)

df = pd.concat([df , new] , axis = 1)

df.drop(["Education" , "Home Onwership" , "Loan Intent"] , axis = 1 , inplace = True)
# print(df)

# print(df.dtypes)

## Apply train test split 

x = df.drop("Loan Status" , axis = 1)
y = df["Loan Status"]

X_train , X_test , y_train , y_test = train_test_split (x , y , test_size = 0.2 , random_state = 42)


## Apply standard scale 

scaler = StandardScaler()

X_train_scalled = scaler.fit_transform(X_train)
X_test_scalled = scaler.transform(X_test)

## Train a model random forest classifier

rfc = RandomForestClassifier()

rfc.fit(X_train_scalled , y_train)
y_pred = rfc.predict(X_test_scalled)

joblib.dump(rfc , "Loan_Approval_rfc_model.pkl")
joblib.dump(scaler , "Loan_Approval_scaler.pkl")

# accurate = accuracy_score(y_test , y_pred)
# confused = confusion_matrix(y_test, y_pred)
## print(accurate)
## print(confused)


