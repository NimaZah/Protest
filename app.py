import json
import os
import pathlib
import urllib.request

import numpy as np
from fastai.tabular.all import *
from flask import Flask, render_template, request

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


path = Path(os.getcwd())
full_path = os.path.join(path, 'protest.pkl')
print("path is:", path)
print("full_path is: ", full_path)
learner = load_learner(full_path)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def prediction():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features)
    final_features = final_features.reshape(1, -1)
    prediction = learner.predict(final_features)
    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':

    app.run(debug=True, use_reloader=False)
