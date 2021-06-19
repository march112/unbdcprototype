import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
import pickle

df = pd.read_csv("text_preprocessed.csv")
df = df[["text", "label"]]
#print(df.head())
df["label"] = df["label"].map({'real': 0, 'fake':1})
#print(df["label"].unique())
df = df.dropna(axis=0, how='any')
X = df["text"].tolist()
y = df["label"].tolist()

X_train, X_test = train_test_split(X, test_size = 0.3, random_state=1)
y_train, y_test = train_test_split(y, test_size=0.3, random_state=1)

cv = CountVectorizer()
cv.fit(X_train)
X_train = cv.transform(X_train) 
X_test = cv.transform(X_test) 
feature_names = cv.get_feature_names()

model = ComplementNB()
model.fit(X_train, y_train)

pickle.dump(cv, open('countercomp.pkl', 'wb'))
pickle.dump(model, open('modelcomp.pkl','wb'))