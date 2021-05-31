import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("datapreprocessed.csv")
df = df[["text", "label"]]
#print(df.head())
df["label"] = df["label"].map({'real': 0, 'fake':1})
#print(df["label"].unique())
df = df.dropna(axis=0, how='any')
X = df["text"].tolist()
y = df["label"].tolist()

x_train, x_test = train_test_split(X, test_size=0.30, random_state=1000)
Y_train, Y_test = train_test_split(y, test_size=0.30, random_state=1000)
vectorizer = CountVectorizer()
vectorizer.fit (x_train)
X_train = vectorizer.transform(x_train)
X_test  = vectorizer.transform(x_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

modelrf = RandomForestClassifier(n_estimators=500, random_state=0) 
modelrf.fit(X_train, Y_train)


pickle.dump(vectorizer, open('rfcounter.pkl', 'wb'))
pickle.dump(modelrf, open('modelrf.pkl','wb'))