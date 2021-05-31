import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pickle

df = pd.read_csv("datapreprocessed.csv")
df = df[["text", "label"]]
#print(df.head())
df["label"] = df["label"].map({'real': 0, 'fake':1})
#print(df["label"].unique())
df = df.dropna(axis=0, how='any')
X = df["text"].tolist()
y = df["label"].tolist()

X_train, X_test = train_test_split(X, test_size=0.30, random_state=1000)
Y_train, Y_test = train_test_split(y, test_size=0.30, random_state=1000)

cv = CountVectorizer()
cv.fit(X_train)
X_train = cv.transform(X_train) # create a sparse matrix
X_test = cv.transform(X_test) # create sparse matrix
feature_names = cv.get_feature_names()

model = SGDClassifier(loss = "log", random_state=0) # loss = "log" for logistic regression
model.fit(X_train,Y_train)

pickle.dump(cv, open('lgcounter.pkl', 'wb'))
pickle.dump(model, open('modellg.pkl','wb'))