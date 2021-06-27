import pickle#Initialize the flask App
import re
import numpy as np

modelnb = pickle.load(open('modelnb.pkl', 'rb'))
counternb = pickle.load(open('counternb.pkl', 'rb'))
modelcm = pickle.load(open('modelcm.pkl', 'rb'))
countercm = pickle.load(open('countercm.pkl', 'rb'))

def cleaning_words(phrase):
  tweet = re.sub(r"http\S+", "", phrase) # remove all URLs
  tweet = re.sub('[^a-zA-z]',' ',tweet) # remove punctuation
  tweet = re.sub(r'@\S+|https?://\S+','', tweet) # remove @ sign
  tweet = tweet.lower() # make all letters lower case
  cleaned_words = []
  cleaned_words.append(tweet)
  return cleaned_words

random_text = "fauci and bill gates want to ban hydroxychloroquine"
random_text = cleaning_words(random_text)
random_text = counternb.transform(random_text)
#random_text = tf.transform(random_text)
print(modelnb.predict(random_text))
print(modelnb.predict_proba(random_text))