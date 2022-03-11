from fastai.tabular.all import *
from flask import Flask, render_template, request
import json
import os
import urllib.request
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

scoring_columns = ["country", "year", "region", "participants"]
# build the path for the trained model
path = Path(os.getcwd())
full_path = os.path.join(path,'trained_model.pkl')

# load the model
learner = load_learner(full_path)

# define the flask app
app = Flask(__name__)

# render home.html page that to serve at localhost that allows user to enter model scoring parameters.
@app.route('/')
def home():
    return render_template('home.html')

# render the results from the model at localhost/results
@app.route('/show-prediction')
def show_prediction():
# load the scoring parameter values into a dictionary indexed by the column names expected by the model.
    score_values_dict = {}
    for column in scoring_columns:
        score_values_dict[column] = request.args.get(column)
    for value in score_values_dict:
        print("value for "+value+" is: "+str(score_values_dict[value]))
        
    # create and load scoring parameters dataframe that will be fed into the pipelines.
    score_df = pd.DataFrame(columns=scoring_columns)
    print("score_df before load is "+str(score_df))
    for col in scoring_columns:
        score_df.at[0, col] = score_values_dict[col]
    # print details about scoring parameters
    print("score_df: ", score_df)
    print("score_df.dtypes: ", score_df.dtypes)
    print("score_df.iloc[0]", score_df.iloc[0])
    print("shape of score_df.iloc[0] is: ", score_df.iloc[0].shape)
    pred_class, pred_idx, outputs = learner.predict(score_df.iloc[0])
    for col in scoring_columns:
        print("value for "+col+" is: ", score_values_dict[col])
    print("The predicted class is", pred_class)

    # get a result string from the value of the model's prediction.
    if pred_class == "1":
        result="Protests will not last long"
    elif pred_class == 2:
        result="Protests will last 3 to 5 days"
    elif pred_class == 3:
        result="Protests will last 5 to 7 days"
    else:
        result="Protests will last 7 to 10 days"
    # build parameter to pass on to show-prediction.html
    prediction = {'prediction_key': predict_string}
    # render the page that will show the prediction
    return(render_template('show-prediction.html', prediction=prediction))


if __name__ == '__main__':

    app.run(debug=True, use_reloader=False)

# write the html file that will be served at localhost.