# Step1: Install libraries.
from fastai.tabular.all import *
from flask import Flask, render_template, request
import json
import os
import urllib.request
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# build the path for the trained model
path = Path(os.getcwd())
full_path = os.path.join(path,'trained_model.pkl')
# load the model
learner = load_learner(full_path)
# set the model in the flask app
app = Flask(__name__)

@app.route("/")
def my_form():
    return render_template('myform.html')

@app.route("/", methods=['POST'])
def my_form_post():
    text = request.form['text']
    # get the prediction from the model
    pred = learner.predict(text)
    # keep the prediction in json format
    prediction = json.dumps(pred[0])
    # return the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run()