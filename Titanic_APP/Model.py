import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder , OrdinalEncoder , OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib

df = pd.read_csv("C:\\Users\\jayap\\OneDrive\\Desktop\\Titanic Dataset For Practice Train.csv")

# print(df)

# print(df.head())

# dfarr = np.array(df.columns)
# print(dfarr)

# print(df.isnull().sum())

# print(df.shape)

# print(df.info)

## Always check duplicates row before drop the column

# print(df.duplicated().sum())

## Drop useless columns 

df.drop(
    ["PassengerId" , "Name" , "Ticket" , "Cabin"] , 
    axis = 1 , 
    inplace = True
)

# print(df)

## Fill NaN values 

# print(df.isnull().sum())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())

# print(df.isnull().sum())

## Check data types 

# print(df.dtypes)


## Now we convert string value into numerical 

## First apply label encoder on sex column 

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])

# print(df.dtypes)
# print(df["Sex"])

## Second apply one hot encoder on embarked column

# print(df["Embarked"].value_counts())

ohe = OneHotEncoder(dtype = int , sparse_output = False)
ohe_scalled = ohe.fit_transform(df[["Embarked"]])
# print(ohe_scalled)

## Check in which values of Embarked onehotencoder is applied
cols = ohe.get_feature_names_out()
# print(cols)

## Convert into new dataframe
new = pd.DataFrame(data = ohe_scalled , columns = cols)
# print(new)

## Join old and new dataframe
df = pd.concat([df , new] , axis = 1)

## Drop original embarked column that contains string value 
df.drop("Embarked" , axis = 1 , inplace = True)

# print(df.dtypes)
# print(df)

## Apply train test split 

x = df.drop("Survived" , axis = 1)
y = df["Survived"]

X_train , X_test , y_train , y_test = train_test_split (x , y , test_size = 0.2 , random_state = 42)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

## Apply standard scale 

scaler = StandardScaler()

X_train_scalled = scaler.fit_transform(X_train)
X_test_scalled = scaler.transform(X_test)


## Train a model logistic regression

lr = LogisticRegression()

lr.fit(X_train_scalled , y_train)
y_pred = lr.predict(X_test_scalled)

# joblib.dump(lr, "Titanic_lr_model.pkl")
# joblib.dump(scaler, "Titanic_scaler.pkl")

accurate = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test, y_pred)
# print(accurate)
# print(confused)

## Train a model random forest classifier

rfc = RandomForestClassifier()

rfc.fit(X_train_scalled , y_train)
y_pred = rfc.predict(X_test_scalled)

# joblib.dump(rfc, "Titanic_rfc_model.pkl")

accurate = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test, y_pred)
# print(accurate)
# print(confused)

## Train a model svc

svc = SVC()

svc.fit(X_train_scalled , y_train)
y_pred = svc.predict(X_test_scalled)

# joblib.dump(svc, "Titanic_svc_model.pkl")

accurate = accuracy_score(y_test , y_pred)
confused = confusion_matrix(y_test, y_pred)
# print(accurate)
# print(confused)


