# app.py

from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

USERNAME = 'user'
PASSWORD = 'password'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('index.html')

@app.route('/recommand_page')
def recommand_page():
    return render_template('recommand.html')

@app.route('/thank_you_page')
def thank_you_page():
    return render_template('thankyou.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            input_array = np.array(features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)

            if prediction[0] == 1:
                return render_template('doctors.html', message='☹️ Heart disease predicted positive. Stay positive take more conscious about your health')
            else:
                return render_template('result.html', message='😊 Heart disease prediction is negative. Happy day ')
        except Exception as e:
            return render_template('result.html', message=f' Error: {e}')

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == USERNAME and password == PASSWORD:
            return redirect(url_for('predict_page'))
        else:
            return "Invalid credentials. Please try again."
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)