from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model_house_price_prediction.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['YearBuilt']
    val2 = request.form['HouseStyle']
    val3 = request.form['OverallQual']
    val4 = request.form['YrSold']
    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)