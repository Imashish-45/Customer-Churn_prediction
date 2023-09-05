from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load the model and encoders
model = pickle.load(open('xgboost_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
one_hot_encoder = pickle.load(open('one_hot_encoder.pkl', 'rb'))

# Create the home page
@app.route('/')
def home():
    return render_template('index.html')

# Create the API to predict churn
@app.route('/predict', methods=['POST'])
def predict():
    # Get the customer inputs
    age = request.form['age']
    subscription_length = request.form['subscription_length']
    monthly_bill = request.form['monthly_bill']
    total_usage_gb = request.form['total_usage_gb']
    gender = request.form['gender']
    location = request.form['location']

    # Prepare the data for the model
    data = pd.DataFrame({
        'Age': [age],
        'Subscription Length (Months)': [subscription_length],
        'Monthly Bill': [monthly_bill],
        'Total Usage GB': [total_usage_gb],
        'Gender': label_encoder.transform([gender]),
        'Location': one_hot_encoder.transform([[location]])
    })

    # Make the prediction
    prediction = model.predict(data)

    # Return the prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
