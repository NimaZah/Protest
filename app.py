import json
import os
import pathlib
import urllib.request

import numpy as np
from fastai.tabular.all import *
from flask import Flask, render_template, request

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# build the path for the trained model
scoring_columns = ['year','participants','country','region']

path = Path(os.getcwd())
full_path = os.path.join(path, 'protest.pkl')
print("path is:", path)
print("full_path is: ", full_path)
learner = load_learner(full_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/show-prediction/')
def show_prediction():
    ''' 
    get the scoring parameters entered in home.html and render show-prediction.html
    '''
    # the scoring parameters are sent to this page as parameters on the URL link from home.html
    # load the scoring parameter values into a dictionary indexed by the column names expected by the model
    score_values_dict = {}
    # bring the URL argument values into a Python dictionary
	# use the scoring parameter values to get a prediction from the model
    for column in scoring_columns:
        # use input from home.html for scoring
        score_values_dict[column] = request.args.get(column)
    for value in score_values_dict:
        print("value for "+value+" is: "+str(score_values_dict[value]))
    # create and load scoring parameters dataframe (containing the scoring parameters)that will be fed into the pipelines
    score_df = pd.DataFrame(columns=scoring_columns)
    print("columns are: ")
    print(scoring_columns)
    for col in scoring_columns:
        score_df.at[0,col] = score_values_dict[col]
    print("score_df is: ")
    print(score_df)
    # ensure columns have the correc types    
    score_df = score_df.astype({"year": int, "participants": int, "country": str, "region": str})
    print("score_df is: ")
    print(score_df)
    # get the model prediction
    prediction = learner.predict(score_df)
    print("prediction is: ", prediction[0])
    # create the return html page
    return render_template('show-prediction.html', prediction_key=str(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)