#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
counter = pickle.load(open('counter.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    words = [x for x in request.form.values()]
    pred_counts = counter.transform(words)
    prediction = model.predict(pred_counts) 
    output = lambda x: "real" if x == 0 else "fake"
    final = output(prediction)
    return render_template('index.html', prediction_text='Our Naive Bayes Model predicts your tweet is :{}'.format(final))

if __name__ == "__main__":
    app.run(debug=True)