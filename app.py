#!/usr/bin/env python

from flask import Flask, request
import pickle
#~ from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
#~ from sklearn.preprocessing import StandardScaler
#~ from sklearn.metrics import precision_score
#~ from sklearn.linear_model import RidgeClassifier

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

global n, pipe
n=0
pipe = pickle.load(open('LR.pickle', 'rb'))
print(pipe)

@app.route("/detect")
def detect():
    global n, pipe
    sms = request.args.get('sms')
    y = pipe.predict([[sms]])
    n+=1
    result= {'sms':sms, 'result':['ham','spam'][int(y[0])], 'n':n}
    return result

#~ app.run()
app.run(host='0.0.0.0', debug=False)
