# from flask import Flask

# app=Flask(__name__)

# @app.route('/')
# def index():
#     return '<h1>Le us see what an amazing job we can do!</h1>'

# @app.route('/user/<name>')
# def user(name):
#     return '<h1>Hello, %s!</h1>' %name

# if __name__=='__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

# Load the model
pickle.dump(trained_model_protests, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/predict',methods=['POST'])
def prediction():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features)
    final_features = final_features.reshape(1,-1)
    prediction = model.predict(final_features)
    return render_template('prediction.html', prediction=prediction)

if __name__ == "__main__":

    app.run(debug=True)