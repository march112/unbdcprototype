import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv("datapreprocessed.csv")
df = df[["text", "label"]]
#print(df.head())
df["label"] = df["label"].map({'real': 0, 'fake':1})
#print(df["label"].unique())
df = df.dropna(axis=0, how='any')
X = df["text"].tolist()
y = df["label"].tolist()

#X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=1)

counter = CountVectorizer()
counter.fit(X)
counts = counter.transform(X)
classifier = MultinomialNB()
classifier.fit(counts, y)
pickle.dump(counter, open('counter.pkl', 'wb'))
pickle.dump(classifier, open('model.pkl','wb'))
# test_counts = counter.transform("Vaccines suck")
# print(classifier.predict(["Vaccines suck"]))