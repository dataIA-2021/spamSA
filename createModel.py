from flask import Flask, request
import pickle
from sklearn.pipeline import Pipeline

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression

X=[["Hello, xxxxxxxxxxxxxxxxxx"], ["EG"], ["xxx"]]
y=[1, 0, 0]

mylen = np.vectorize(len)
tr = FunctionTransformer(mylen)

pipe = Pipeline(steps=[
   ('len', tr),
   ('clf', LogisticRegression())])

pipe.fit(X, y)
pickle.dump(pipe, open('LR.pickle', 'wb'))
