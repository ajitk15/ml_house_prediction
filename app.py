import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np
import bz2file as bz2
import os

#Initiate Flask
app=Flask(__name__)

#Load model
scaler=pickle.load(open('StandardScaler.pkl','rb'))
model=pickle.load(open('house_pred_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    new_pred = model.predict(new_data)
    print(new_pred[0])
    return jsonify(new_pred[0])
    
@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(np.array(data).reshape(1, -1))
    new_data = scaler.transform(np.array(data).reshape(1, -1))
    new_pred = model.predict(new_data)
    print(new_pred[0])
    return render_template('home.html', prediction_text=f"Esitmated Home price {new_pred[0]}")

if __name__ == "__main__":
    app.run(debug=True)
