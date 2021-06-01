#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
modelmn = pickle.load(open('model.pkl', 'rb'))
countermn = pickle.load(open('counter.pkl', 'rb'))
modelcm = pickle.load(open('modelcomp.pkl', 'rb'))
countercm = pickle.load(open('countercomp.pkl', 'rb'))
# modelrf = pickle.load(open('modelrf.pkl', 'rb'))
# counterrf = pickle.load(open('rfcounter.pkl', 'rb'))
modelknn = pickle.load(open('modelknn.pkl', 'rb'))
counterknn = pickle.load(open('knncounter.pkl', 'rb'))
modelsvm = pickle.load(open('modelsvm.pkl', 'rb'))
countersvm = pickle.load(open('svmcounter.pkl', 'rb'))
modellg = pickle.load(open('modellg.pkl', 'rb'))
counterlg = pickle.load(open('lgcounter.pkl', 'rb'))

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
    # pred3_counts = counterrf.transform(words)
    # prediction3 = modelrf.predict(pred3_counts)
    pred4_counts = counterknn.transform(words)
    prediction4 = modelknn.predict(pred4_counts)    
    pred5_counts = countersvm.transform(words)
    prediction5 = modelsvm.predict(pred5_counts)    
    pred6_counts = counterlg.transform(words)
    prediction6 = modellg.predict(pred6_counts)
    output = lambda x: "Real" if x == 0 else "Fake"
    final1 = output(prediction1)
    final2 = output(prediction2)
    # final3 = output(prediction3)
    final4 = output(prediction4)
    final5 = output(prediction5)
    final6 = output(prediction6)
    return render_template('index.html', prediction_text=' Our Naive Bayes Multinomial Model predicts your tweet is {0}. \nOur Naive Bayes Complement Model predicts your tweet is {1}. \n Our K Nearest Neighbours Model predicts your tweet is {2}. Our Support Vector Machine Model predicts your tweet is {3}. Our Logistic Regression model predicts your tweet is {4}'.format(final1, final2, final4, final5, final6))

if __name__ == "__main__":
    app.run(debug=True)