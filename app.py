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

@app.route("/detect")
def detect():
    sms = request.args.get('sms')
    pipe = pickle.load(open('LR.pickle', 'rb'))
    print(pipe)
    y = pipe.predict([[sms]])
    print(y)
    result= {'sms':sms, 'result':['ham','spam'][int(y[0])]}
    return result

app.run()