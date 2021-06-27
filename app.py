#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
import re
app = Flask(__name__)
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

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    words = [x for x in request.form.values()]
    # #For rendering results on HTML GUI
    words = cleaning_words(words[0])
    pred1_counts = counternb.transform(words)
    pred2_counts = countercm.transform(words)
    prediction1 = modelnb.predict(pred1_counts)
    prediction2 = modelcm.predict(pred2_counts) 
    output = lambda x: "Real" if x == 0 else "Fake"
    final1 = output(prediction1)
    final2 = output(prediction2)
    # # final3 = output(prediction3)
    # # final4 = output(prediction4)
    # # final5 = output(prediction5)
    # # final6 = output(prediction6)
    #return render_template('index.html', prediction_text=words)
    return render_template('index.html', prediction_text=' Our Naive Bayes Multinomial Model predicts your tweet is {0}. \nOur Naive Bayes Complement Model predicts your tweet is {1}'.format(final1, final2))
    # # \n Our K Nearest Neighbours Model predicts your tweet is {2}. Our Support Vector Machine Model predicts your tweet is {3}. Our Logistic Regression model predicts your tweet is {4}. Our Random Forest Model predicts your tweet is {5}'.format(final1, final2, final4, final5, final6, final3))

if __name__ == "__main__":
    app.run(debug=True)