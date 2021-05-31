#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
modelmn = pickle.load(open('model.pkl', 'rb'))
countermn = pickle.load(open('counter.pkl', 'rb'))
modelcm = pickle.load(open('modelcomp.pkl', 'rb'))
countercm = pickle.load(open('countercomp.pkl', 'rb'))
modelrf = pickle.load(open('modelrf.pkl', 'rb'))
counterrf = pickle.load(open('rfcounter.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    words = [x for x in request.form.values()]
    pred1_counts = countermn.transform(words)
    prediction1 = modelmn.predict(pred1_counts)
    pred2_counts = countercm.transform(words)
    prediction2 = modelcm.predict(pred2_counts) 
    pred3_counts = counterrf.transform(words)
    prediction3 = modelrf.predict(pred3_counts)  
    output = lambda x: "Real" if x == 0 else "Fake"
    final1 = output(prediction1)
    final2 = output(prediction2)
    final3 = output(prediction3)
    return render_template('index.html', prediction_text=' Our Naive Bayes Multinomial Model predicts your tweet is {0}. \nOur Naive Bayes Complement Model predicts your tweet is {1}. \nRandom Forest Model predicts your tweet is {2}.'.format(final1, final2, final3))

if __name__ == "__main__":
    app.run(debug=True)